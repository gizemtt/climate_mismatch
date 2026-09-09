"""Fixed-reference minimax-regret healthcare planning: ONE standard robust MIP.

READING MAP
  get_inputs                 : read knee y files and select a row per target.
  prepare_benchmarks         : reuse compatible table b or recompute ONCE.
  build_fixed_regret_model   : min eta, f1[c]-b[c]<=eta, f2[c]<=L[c].
  evaluate_final_portfolio   : final fixed-y Q1 and optional matched benchmark.
  run                        : orchestrate setup, one robust solve, reporting.

No nested solves during the robust solve. No custom callback, lazy cuts,
one-hot depth encoding, or portfolio enumeration. Gurobi uses its normal MIP
solver. The fixed-reference objective differs from matched-frontier regret.

INPUTS (relative to project root)
  output/cross_eval/summary_fixed_policy_eval.csv
    Filter Budget; for each target_scenario c choose largest fixed_f2.
    L[c]=that row's fixed_f2; candidate b[c]=its benchmark_f1.
  output/S_245 (or S_main/S_perc)/B{B}_operationMetrics/epsilon/eps_*/y_positive.csv
    Fall back to B{B}; also accept eps_* directly below budget directory.
    Select middle numeric epsilon folder, as in the existing workflow.
    Missing facility rows mean y=0. run_meta.csv is never read.
  src.config.Config and src.data_loader.load_data
    Existing project data and K-nearest receiver map (including DUMMY).

For each target climate c, L[c] is the selected cross-evaluation row's
fixed_f2. If that same row contains a valid benchmark_f1, it is used directly
as b[c] during robust-model preparation; no benchmark optimization is run and
benchmark_eps is not checked. A benchmark is solved before the robust model
only when benchmark_f1 is missing or invalid. A small numerical tolerance is
added to L[c] when it is imposed as an f2 upper bound.

After the robust portfolio is selected, the final fixed-policy evaluations are
completed and the matched-frontier benchmark is solved post hoc at the final
portfolio's achieved Q2. These post-processing solves do not affect the robust
portfolio.

The --time_limit parameter applies ONLY to the single robust minimax-regret
optimization. Input loading, reference-benchmark preparation, model building,
file export, fixed-policy evaluation, and optional matched-frontier
post-processing are outside this time limit. If the robust MIP reaches the
time limit with an incumbent, that best available incumbent is saved together
with Gurobi's incumbent objective, best bound, and reported MIP gap, and ALL
post-processing is then completed using that incumbent.

OUTPUTS are plain CSV, not archives. robust_portfolio.csv contains every J,
including y=0. Operational decisions are stored in separate climate files.
The optional matched-frontier benchmarks are solved only AFTER y is fixed.
fixed_benchmarks.csv stores L, b and source-row metadata together per climate.
All solves share --mip_gap (default 1e-5, i.e. 0.001% relative gap); only the
robust minimax-regret solve has a time limit.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
from typing import Any, Mapping, Iterable, List, Sequence, Tuple
import gurobipy as gp
from gurobipy import GRB
from src.config import Config
from src.data_loader import load_data

CLIMATES={'ssp245':'S_245','main':'S_main','perc':'S_perc'}

def normalize_id(value: Any) -> str:
    """Create a stable facility identifier across climate data sets."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(round(float(text))))
    except (TypeError, ValueError):
        return text

def mapping_from(data: Mapping[str, Any], names: Sequence[str], purpose: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(
        f"Could not locate {purpose} in loaded data. Tried keys {list(names)}; "
        f"available keys are {sorted(map(str, data.keys()))}."
    )

def parameter_value(parameter: Any, *indices: Any) -> float:
    if callable(parameter):
        return float(parameter(*indices))
    if len(indices) == 1:
        return float(parameter[indices[0]])
    try:
        return float(parameter[tuple(indices)])
    except (KeyError, TypeError):
        value = parameter
        for index in indices:
            value = value[index]
        return float(value)

def receiver_capacity(data: Mapping[str, Any]) -> Any:
    for name in ("A_avail", "A", "Avail", "available_capacity", "A_j"):
        if name in data:
            return data[name]
    if "O" in data and "D" in data:
        return {
            j: parameter_value(data["O"], j) - parameter_value(data["D"], j)
            for j in data["J"]
        }
    raise KeyError(
        "Receiver capacity not found; expected A_avail, A, or both O and D."
    )

def has_parameter_index(parameter: Any, index: Any) -> bool:
    if callable(parameter):
        try:
            parameter(index)
            return True
        except (KeyError, TypeError, ValueError):
            return False
    try:
        parameter[index]
        return True
    except (KeyError, TypeError, IndexError):
        return False

def evacuation_arcs(data: Mapping[str, Any], scenario: Any) -> List[Tuple[Any, Any]]:
    """Use stored arcs when available; otherwise apply the manuscript definition."""
    # This is the final scenario- and sender-specific K-nearest-neighbor map
    # produced by the project's data loader. It includes the DUMMY receiver
    # when required to preserve complete recourse.
    if "Kmap_s_j_listk" in data:
        neighbor_map = data["Kmap_s_j_listk"]
        arcs: List[Tuple[Any, Any]] = []
        for sender in data["Jsend"]:
            receivers: Iterable[Any] = []
            if isinstance(neighbor_map, Mapping):
                receivers = neighbor_map.get((scenario, sender), [])
                if not receivers:
                    scenario_map = neighbor_map.get(scenario, {})
                    if isinstance(scenario_map, Mapping):
                        receivers = scenario_map.get(sender, [])
            for receiver in receivers:
                arcs.append((sender, receiver))
        if arcs:
            return arcs

    for key in ("A_s", "Arcs", "arcs", "evac_arcs"):
        if key not in data:
            continue
        raw = data[key]
        arcs = raw.get(scenario, []) if isinstance(raw, Mapping) else raw
        return [(j, k) for j, k in arcs]

    capacities = receiver_capacity(data)
    flood = data["F"]
    senders = list(data["Jsend"])
    receivers = [
        k
        for k in data["J"]
        if parameter_value(flood, scenario, k) <= 1e-9
        and parameter_value(capacities, k) > 1e-9
    ]
    costs = data["C_E"]
    arcs: List[Tuple[Any, Any]] = []
    for j in senders:
        for k in receivers:
            if j == k:
                continue
            try:
                parameter_value(costs, j, k)
            except (KeyError, TypeError, ValueError):
                continue
            arcs.append((j, k))
    return arcs

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save_csv(path, records):
    if not records: return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

def integer(v):
    if not math.isfinite(v) or v < 0 or abs(v-round(v))>1e-7:
        raise ValueError(f'Expected nonnegative integer datum, got {v}')
    return int(round(v))

def get_inputs(args,B,out):
    selected={}; log=[]
    for c,folder in CLIMATES.items():
        climate_root=Path(args.base_output_dir)/folder
        budget_roots=(
            climate_root/f'B{B}_operationMetrics',
            climate_root/f'B{B}',
        )
        searched=[]; root=None; dirs=[]
        for budget_root in budget_roots:
            for candidate in (budget_root/'epsilon', budget_root):
                searched.append(str(candidate))
                candidate_dirs=sorted(
                    (p for p in candidate.glob('eps_*') if p.is_dir()),
                    key=lambda p:float(p.name[4:]),
                )
                if candidate_dirs:
                    root=candidate; dirs=candidate_dirs; break
            if dirs: break
        if not dirs:
            raise FileNotFoundError(
                'No eps_* folders found. Searched: '+', '.join(searched)
            )
        p=dirs[len(dirs)//2]; selected[c]={}
        for r in rows(p/'y_positive.csv'):
            j=normalize_id(r['j']); v=integer(float(r['y']))
            if j in selected[c] and selected[c][j]!=v: raise ValueError(f'Duplicate {j}')
            selected[c][j]=v
        log.append(dict(climate=c,budget=B,epsilon=float(p.name[4:]),
                        epsilon_root=str(root),y_file=str(p/'y_positive.csv')))
    table_path=Path(args.cross_eval_csv) if args.cross_eval_csv else Path(args.base_output_dir)/'cross_eval'/'summary_fixed_policy_eval.csv'
    table=rows(table_path)
    table=[r for r in table if abs(float(r.get('Budget',r.get('budget','nan')))-B)<0.5]
    if not table: raise ValueError(f'No cross-evaluation rows for {B}')
    # Budget alone determines L. Validation prevents mixing different experiments.
    pairs={(a,c) for a in CLIMATES for c in CLIMATES if a!=c}
    found=[(r['source_scenario'],r['target_scenario']) for r in table]
    if len(table)!=6 or set(found)!=pairs:
        raise ValueError('Expected exactly six distinct off-diagonal evaluations per budget.')
    for r in table:
        eps=next(t['epsilon'] for t in log if t['climate']==r['source_scenario'])
        if not math.isclose(float(r['source_eps']),eps,rel_tol=0,abs_tol=1e-5):
            raise ValueError('Cross-evaluation source_eps does not match selected warm-start folder.')
    vals=[float(r['fixed_f2']) for r in table]
    if not all(math.isfinite(v) and v>=0 for v in vals): raise ValueError('Invalid fixed_f2')
    # Exactly one row per TARGET climate. L and b always come from the SAME row.
    references={}
    for c in CLIMATES:
        incoming=[r for r in table if r['target_scenario']==c]
        chosen=max(incoming,key=lambda r:float(r['fixed_f2']))
        table_b=optional_number(chosen.get('benchmark_f1'))
        if table_b is not None and table_b < 0:
            table_b=None
        references[c]=dict(
            budget=B,
            climate=c,
            L=float(chosen['fixed_f2']),
            table_b=table_b,
            source_scenario=chosen['source_scenario'],
            source_table=str(table_path),
            source_eps=float(chosen['source_eps']),
            row_fixed_f1=float(chosen['fixed_f1']),
        )
    save_csv(out/'selection_log.csv',log)
    return selected,references,table

def params(m,args,time_limit=None):
    """Apply common solver settings; TimeLimit is used only when explicitly supplied."""
    m.Params.OutputFlag=int(not args.quiet)
    m.Params.MIPGap=args.mip_gap
    m.Params.MIPGapAbs=0
    if args.threads:
        m.Params.Threads=args.threads
    # Every model is unlimited by default. Only the robust solve passes a limit.
    m.Params.TimeLimit=GRB.INFINITY if time_limit is None else max(.01,float(time_limit))

def shared_data(data):
    bounds={}; costs={}
    for c,d in data.items():
        cost=mapping_from(d,('C_H','CH','hardening_cost'),'hardening cost')
        for j in d['Jsend']:
            k=normalize_id(j)
            h=max(integer(parameter_value(d['F'],s,j)) for s in d['S'])
            bounds[k]=max(bounds.get(k,0),h)
            v=parameter_value(cost,j)
            if v<0: raise ValueError('Negative hardening cost')
            if k in costs and not math.isclose(costs[k],v,rel_tol=1e-8):
                raise ValueError(f'Inconsistent climate costs at {k}')
            costs[k]=v
    return bounds,costs

def build(data,bounds,costs,B,L,args,name,env=None):
    m=gp.Model(name,env=env)
    y={j:m.addVar(vtype=GRB.INTEGER,ub=h,name=f'y[{j}]') for j,h in bounds.items()}
    m.addConstr(gp.quicksum(costs[j]*v for j,v in y.items())<=B,name='budget')
    f1={}; f2={}; limits={}
    for c,d in data.items():
        demand=mapping_from(d,('D_inpt','D','demand','evac_demand'),'demand')
        cap=receiver_capacity(d); a=[]; b=[]
        for s in d['S']:
            p=parameter_value(d['P'],s)
            if p<0: raise ValueError('Negative probability')
            outgoing={}; incoming={}
            for j,k in evacuation_arcs(d,s):
                q=m.addVar(vtype=GRB.INTEGER,name=f'q[{c},{s},{j},{k}]')
                outgoing.setdefault(j,[]).append(q); incoming.setdefault(k,[]).append(q)
                a.append(p*parameter_value(d['C_E'],j,k)*q)
            for k,qq in incoming.items():
                if has_parameter_index(cap,k):
                    m.addConstr(gp.quicksum(qq)<=integer(parameter_value(cap,k)))
                elif str(k).upper()!='DUMMY':
                    raise ValueError(f'Missing capacity for non-DUMMY receiver {k}')
            for j in d['Jsend']:
                F=integer(parameter_value(d['F'],s,j)); yy=y[normalize_id(j)]
                g=m.addVar(vtype=GRB.BINARY,name=f'gamma[{c},{s},{j}]')
                z=m.addVar(ub=F,name=f'z[{c},{s},{j}]')
                # Exact physical exposure, including in the search master.
                m.addGenConstrIndicator(g,False,yy>=F)
                m.addGenConstrIndicator(g,False,z==0)
                m.addGenConstrIndicator(g,True,yy<=F-1)
                m.addGenConstrIndicator(g,True,z==F-yy)
                m.addConstr(gp.quicksum(outgoing.get(j,[]))==integer(parameter_value(demand,j))*g)
                a.append(p*parameter_value(d['R'],j)*z)
                b.append(p*parameter_value(d['V'],j)*z)
        f1[c]=gp.quicksum(a); f2[c]=gp.quicksum(b)
        rhs=(L[c] if isinstance(L,Mapping) else L) + args.reference_tol
        limits[c]=m.addConstr(f2[c]<=rhs,name=f'L[{c}]')
    m.setObjective(gp.quicksum(f1.values()),GRB.MINIMIZE)
    m.update()
    return m,y,f1,f2,limits

def starts(m,y,portfolios):
    m.NumStart=len(portfolios)
    for n,plan in enumerate(portfolios):
        m.Params.StartNumber=n
        for j,v in y.items(): v.Start=plan.get(j,0)
    m.update()

def q2(d,plan):
    return sum(parameter_value(d['P'],s)*parameter_value(d['V'],j)*
               max(0,parameter_value(d['F'],s,j)-plan.get(normalize_id(j),0))
               for s in d['S'] for j in d['Jsend'])

def optional_number(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):
        return None

def finite_nonnegative(value,name):
    v=optional_number(value)
    if v is None or v<0: raise ValueError(f'Invalid {name}: {value}')
    return v

def solve_standard(m,args,time_limit=None):
    """Run one standard Gurobi solve.

    When time_limit is None, the solve is not time-limited and terminates by the
    requested MIP gap (or another solver termination condition). A finite
    time_limit is used only for the robust minimax-regret optimization.
    """
    deadline=None if time_limit is None else time.monotonic()+float(time_limit)
    params(m,args,time_limit)
    m.optimize()

    if m.Status==GRB.INF_OR_UNBD:
        m.Params.DualReductions=0
        if deadline is None:
            params(m,args,None)
            m.optimize()
        else:
            remaining=deadline-time.monotonic()
            if remaining>0:
                params(m,args,remaining)
                m.optimize()

    if not m.SolCount:
        return None

    bound=float(m.ObjBound)
    return dict(
        objective=float(m.ObjVal),
        lower=bound if math.isfinite(bound) else 0.,
        mip_gap=float(m.MIPGap),
        status=int(m.Status),
        sol_count=int(m.SolCount),
    )

def snapshot(m):
    return m.getAttr('X',m.getVars())

def set_complete_start(m,values):
    m.NumStart=1; m.Params.StartNumber=0
    m.setAttr('Start',m.getVars(),values)
    m.update()

def prepare_benchmarks(data,references,selected,costs,args,B,out):
    """
    Prepare the fixed reference constants used by the robust model.

    For each target climate c:
      * L[c] is ALWAYS the selected cross-evaluation row's fixed_f2.
      * If benchmark_f1 is available in that same row, use it directly as b[c].
        No benchmark_eps/fixed_f2 comparison is performed.
      * Only if benchmark_f1 is missing/invalid do we solve one benchmark model
        at f2 <= L[c] + reference_tol.

    Thus, existing cross-evaluation data are reused whenever available.
    """
    records=[]
    for c,d in data.items():
        ref=references[c]
        L=ref['L']
        table_b=ref.get('table_b')

        if table_b is not None and math.isfinite(table_b) and table_b >= 0:
            ref.update(
                b=float(table_b),
                b_lower=None,
                benchmark_mip_gap=None,
                benchmark_status=None,
                benchmark_origin='cross_eval_benchmark_f1',
            )
            print(
                f'[REFERENCE] {c}: L=fixed_f2={L:,.6f}, '
                f'b=benchmark_f1={ref["b"]:,.6f} (reused from table)',
                flush=True,
            )
        else:
            hb,cb=shared_data({c:d})
            m,x,f1,f2,_=build(
                {c:d},hb,cb,B,L,args,f'fixed_reference_missing_benchmark_{c}'
            )
            try:
                starts(
                    m,x,
                    [{j:min(p.get(j,0),hb[j]) for j in hb}
                     for p in selected.values()]
                )
                result=solve_standard(m,args)
                if result is None:
                    raise RuntimeError(
                        f'benchmark_f1 is unavailable for {c} and the fallback '
                        f'benchmark optimization returned no incumbent; '
                        f'status={m.Status}.'
                    )
                ref.update(
                    b=result['objective'],
                    b_lower=result['lower'],
                    benchmark_mip_gap=result['mip_gap'],
                    benchmark_status=result['status'],
                    benchmark_origin='optimized_only_because_benchmark_f1_missing',
                )
                print(
                    f'[REFERENCE] {c}: L=fixed_f2={L:,.6f}, '
                    f'b={ref["b"]:,.6f} (benchmark_f1 missing; optimized)',
                    flush=True,
                )
            finally:
                m.dispose()

        records.append(dict(ref))
        save_csv(out/'fixed_benchmarks.csv',records)

    return references

def build_fixed_regret_model(data,bounds,costs,B,references,args):
    """
    THE ROBUST OPTIMIZATION -- one ordinary mixed-integer linear model.

       minimize eta
       subject to f1[c](y,z[c],q[c]) - b[c] <= eta   for every target c
                  f2[c](y,z[c]) <= L[c]             for every target c
                  sum_j CH[j]*y[j] <= B
                  physical failure/evacuation/capacity constraints
                  integer hardening, binary failure, integer patient flows
                  eta >= 0.

    b[c] and L[c] are FIXED before optimize(). They NEVER change when y changes.
    y is shared across all climate blocks. Each block has separate recourse.
    build() initially sets an economic objective; we REPLACE it below with eta.
    With approximate/reported b values, raw f1-b can be slightly negative;
    eta>=0 minimizes the nonnegative maximum excess over the fixed references.
    This is NOT the endogenous matched-frontier-regret objective.

    Using recourse variables in f1 is a valid extended formulation: if minimum
    recourse cost <= b+eta, an optimal recourse assignment witnesses feasibility.
    Nonbinding climate recourse may nevertheless be suboptimal in the returned
    incumbent, which is why final fixed-y evaluations are reported separately.
    """
    L={c:r['L'] for c,r in references.items()}
    m,y,f1,f2,_=build(data,bounds,costs,B,L,args,'fixed_reference_minimax_regret')
    eta=m.addVar(lb=0,name='eta')
    for c in data:
        m.addConstr(f1[c]-references[c]['b']<=eta,name=f'fixed_regret[{c}]')
    m.setObjective(eta,GRB.MINIMIZE)
    m.update()
    return m,y,eta,f1,f2

def export_operations(path,m,climate,all_climates=False):
    """Write uncompressed operational variables, including zero values."""
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['climate','variable','value'])
        for v in m.getVars():
            if v.VarName.startswith(('q[','gamma[','z[')):
                c=v.VarName.split('[',1)[1].split(',',1)[0] if all_climates else climate
                w.writerow([c,v.VarName,v.X])

def add_regret_fields(row):
    b=row['b_used']; lo=row['Q1_lower']; hi=row['Q1_upper']
    row['fixed_reference_regret_estimate']=hi-b
    row['fixed_reference_regret_lower']=lo-b
    row['fixed_reference_regret_upper']=hi-b
    row['fixed_reference_percent_regret']=100*(hi-b)/b if b>1e-9 else None
    row['percentage_status']='defined' if b>1e-9 else 'undefined_zero_benchmark'

def evaluate_final_portfolio(data,plan,references,args,B,out,performance):
    """
    AFTER the robust solve, optimize recourse with the final common y fixed.
    Then optionally calculate the Table-1 matched-frontier benchmark at its
    achieved Q2. Those extra benchmarks DO NOT affect the selected portfolio.
    Complete fixed-recourse solutions warm-start these post-hoc benchmarks.
    Partial/time-limited results retain bounds and explicit statuses.
    """
    models={}
    try:
        for c,d in data.items():
            hb,cb=shared_data({c:d})
            m,x,f1,f2,eps=build({c:d},hb,cb,B,references[c]['L'],args,f'final_fixed_{c}')
            models[c]=(m,x,f1,f2,eps,hb)
            for j,v in x.items(): v.LB=v.UB=min(plan.get(j,0),hb[j])
            old=performance[c]
            # Seed from the full robust recourse CSV would require parsing millions
            # of rows. Hardening is fixed here; Gurobi solves the residual network.
            result=solve_standard(m,args)
            if result:
                # Both robust recourse and fixed evaluation are feasible witnesses.
                upper=min(old['Q1_upper'],result['objective'])
                old.update(Q1_upper=upper,Q1_lower=max(0,result['lower']),
                    fixed_status=result['status'],fixed_mip_gap=result['mip_gap'],
                    Q1_status='fixed_policy_solved')
                if result['objective']<=old['model_incumbent_f1']+1e-7:
                    export_operations(out/f'operational_decisions_{c}.csv',m,c)
                old['Q1']=upper
                add_regret_fields(old)
            save_csv(out/'climate_performance.csv',list(performance.values()))
        if not args.matched_evaluation: return
        for c,(m,x,f1,f2,eps,hb) in models.items():
            if not m.SolCount:
                continue
            start=snapshot(m)
            for j,v in x.items(): v.LB=0;v.UB=hb[j]
            eps[c].RHS=performance[c]['Q2'] + args.reference_tol
            set_complete_start(m,start)
            result=solve_standard(m,args)
            if result:
                row=performance[c]; mb=result['objective']; ml=max(0,result['lower'])
                row.update(matched_benchmark=mb,matched_benchmark_lower=ml,
                    matched_benchmark_f2=float(f2[c].getValue()),
                    matched_benchmark_mip_gap=result['mip_gap'],
                    matched_status=result['status'],
                    matched_regret_estimate=row['Q1_upper']-mb,
                    matched_regret_lower=max(0,row['Q1_lower']-mb),
                    matched_regret_upper=max(0,row['Q1_upper']-ml),
                    matched_percent_regret=100*(row['Q1_upper']-mb)/mb if mb>1e-9 else None)
            save_csv(out/'climate_performance.csv',list(performance.values()))
    finally:
        for entry in models.values(): entry[0].dispose()

def reference_comparison(table,references,args,out):
    """Historical off-diagonal observations only; not newly optimized Q1 values."""
    result=[]
    for r in table:
        c=r['target_scenario'];b=references[c]['b'];Q1=float(r['fixed_f1']);Q2=float(r['fixed_f2'])
        result.append(dict(source=r['source_scenario'],target=c,
            historical_Q1=Q1,historical_Q2=Q2,L=references[c]['L'],b_used=b,
            meets_target_L=Q2<=references[c]['L']+args.reference_tol,
            fixed_reference_regret=Q1-b,
            fixed_reference_percent_regret=100*(Q1-b)/b if b>1e-9 else None,
            historical_matched_benchmark=float(r['benchmark_f1'])))
    save_csv(out/'reference_cross_climate_comparison.csv',result)

def run(args,B):
    out=Path(args.output_dir)/f'B{B}';out.mkdir(parents=True,exist_ok=True)
    if (out/'run_summary.json').exists():
        raise FileExistsError(f'Results already exist in {out}; choose another --output_dir.')
    selected,references,table=get_inputs(args,B,out)
    (out/'run_configuration.json').write_text(json.dumps(vars(args),indent=2))
    data={c:load_data(Config(scenario_set=c,solution_method='benders',objective_mode='eps',
          Budget=B,K=args.K,mip_gap=args.mip_gap,verbose=not args.quiet)) for c in CLIMATES}
    bounds,costs=shared_data(data)
    for c,plan in selected.items():
        if any(v and j not in bounds for j,v in plan.items()):
            raise ValueError(f'Unknown positive warm-start facility in {c}')
        if any(v>bounds.get(j,0) for j,v in plan.items()):
            raise ValueError(f'Warm-start hardening exceeds shared bounds in {c}')
    started=time.monotonic()
    summary=dict(
        budget=B,
        status='setup',
        method='fixed_reference_minimax_regret',
        robust_time_limit_seconds=args.time_limit,
        time_limit_scope='robust_optimization_only',
    )
    m=None;performance={}
    try:
        prepare_benchmarks(data,references,selected,costs,args,B,out)
        reference_comparison(table,references,args,out)
        # Record each reference portfolio's diagonal AND cross-climate disruption.
        startchecks=[]
        for source,plan in selected.items():
            spend=sum(costs[j]*plan.get(j,0) for j in bounds)
            for c,d in data.items():
                u=q2(d,plan)
                startchecks.append(dict(source=source,target=c,Q2=u,L=references[c]['L'],
                    disruption_feasible=u<=references[c]['L']+args.reference_tol,hardening_spend=spend,
                    budget_feasible=spend<=B+1e-5))
        save_csv(out/'warm_start_checks.csv',startchecks)
        m,y,eta,f1,f2=build_fixed_regret_model(data,bounds,costs,B,references,args)
        starts(m,y,list(selected.values()))
        if args.write_lp: m.write(str(out/'robust_model.lp'))
        # --time_limit starts HERE and applies only to this robust MIP solve.
        robust_started=time.monotonic()
        result=solve_standard(m,args,args.time_limit)
        robust_elapsed=time.monotonic()-robust_started
        summary.update(
            robust_status=int(m.Status),
            robust_solve_seconds=robust_elapsed,
            status='no_robust_incumbent' if result is None else 'incumbent',
        )
        if result is None: return
        plan={j:int(round(v.X)) for j,v in y.items()}
        ids=sorted({normalize_id(j) for d in data.values() for j in d['J']}|set(bounds))
        save_csv(out/'robust_portfolio.csv',[dict(j=j,y=plan.get(j,0)) for j in ids])
        # Save every raw robust operational decision before any evaluation can time out.
        export_operations(out/'robust_model_operations.csv',m,'',all_climates=True)
        for c,d in data.items():
            value=float(f1[c].getValue())
            row=dict(climate=c,L=references[c]['L'],b_used=references[c]['b'],
                benchmark_origin=references[c]['benchmark_origin'],
                Q2=q2(d,plan),model_incumbent_f1=value,Q1=None,Q1_lower=0.,Q1_upper=value,
                Q1_status='not_evaluated_feasible_upper_bound_only',
                fixed_status=None,fixed_mip_gap=None,
                matched_benchmark=None,matched_benchmark_lower=None,
                matched_benchmark_f2=None,matched_benchmark_mip_gap=None,matched_status=None,
                matched_regret_estimate=None,matched_regret_lower=None,matched_regret_upper=None,
                matched_percent_regret=None)
            add_regret_fields(row);performance[c]=row
        save_csv(out/'climate_performance.csv',list(performance.values()))
        summary.update(robust_eta=result['objective'],robust_lower_bound=result['lower'],
            robust_mip_gap=result['mip_gap'],hardening_spend=sum(costs[j]*v for j,v in plan.items()),
            hardened_facilities=sum(v>0 for v in plan.values()))
        # The robust model need not occupy memory during final evaluations.
        m.dispose();m=None
        evaluate_final_portfolio(data,plan,references,args,B,out,performance)
        ub=max(0,max(r['Q1_upper']-r['b_used'] for r in performance.values()))
        lb=max(0,result['lower'])
        summary.update(
            status='robust_time_limit_postprocessing_completed'
                   if result['status']==GRB.TIME_LIMIT else 'completed',
            fixed_constant_regret_upper_bound=ub,
            fixed_constant_regret_lower_bound=lb,
            fixed_constant_absolute_gap=max(0,ub-lb),
            fixed_constant_relative_gap=max(0,ub-lb)/ub if ub>0 else 0.,
            evaluation_complete=all(r['Q1'] is not None for r in performance.values()),
            matched_evaluation_complete=(
                all(r['matched_status'] is not None for r in performance.values())
                if args.matched_evaluation else None
            ),
            guarantee='Bounds apply to the fixed b values used; table/benchmark uncertainty is separate.',
        )
    except Exception as exc:
        summary.update(status='error',error=repr(exc));raise
    finally:
        summary['elapsed_seconds']=time.monotonic()-started
        if m is not None:m.dispose()
        save_csv(out/'run_summary.csv',[summary])
        (out/'run_summary.json').write_text(json.dumps(summary,indent=2))
        (out/'results_summary.txt').write_text('\n'.join(f'{k}: {v}' for k,v in summary.items()))
        print(f'[DONE] {out}: {summary["status"]}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base_output_dir',default='output')
    p.add_argument('--cross_eval_csv',default=None,help='Optional override of cross-evaluation summary path.')
    p.add_argument('--output_dir',default='output/minimax_regret_fixed_reference')
    p.add_argument('--budgets',nargs='+',type=int,default=[50000000,300000000,600000000])
    p.add_argument('--K',type=int,default=110)
    p.add_argument('--mip_gap',type=float,default=1e-5,
                   help='Relative optimality gap for ALL solves (default: 1e-5 = 0.001%).')
    p.add_argument(
        '--reference_tol',type=float,default=1e-6,
        help='Small absolute tolerance added to fixed_f2/Q2 RHS values (default: 1e-6).'
    )
    p.add_argument(
        '--time_limit',type=float,default=3600.,
        help='Time limit in seconds for the robust minimax-regret optimization ONLY. '
             'Benchmark preparation and all post-processing are not time-limited.'
    )
    p.add_argument('--matched_evaluation',action=argparse.BooleanOptionalAction,default=True,
                   help='Post-hoc matched-frontier benchmark for the final portfolio.')
    p.add_argument('--threads',type=int,default=None)
    p.add_argument('--quiet',action='store_true');p.add_argument('--write_lp',action='store_true')
    args=p.parse_args()
    if not math.isfinite(args.mip_gap) or args.mip_gap<0:
        p.error('MIP gap must be finite and nonnegative.')
    if not math.isfinite(args.reference_tol) or args.reference_tol < 0:
        p.error('Reference tolerance must be finite and nonnegative.')
    if not math.isfinite(args.time_limit) or args.time_limit<=0:
        p.error('Time limit must be finite and strictly positive.')
    for B in args.budgets:run(args,B)

if __name__=='__main__':main()
