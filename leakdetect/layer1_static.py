"""Layer 1 - static pipeline graph analysis.

Three input forms, all lowered to one networkx DiGraph and checked by the same rules:
  * Python source (AST data-flow DAG, gives line numbers)   -> analyze_source / analyze_file
  * explicit step list [(name, callable, op_type), ...]      -> analyze_steps
  * an sklearn Pipeline / ColumnTransformer                  -> analyze_pipeline

Graph: node attr `kind` in {var, split, cv_split, cv_eval, fit, fit_transform, fit_resample, stat, transform};
an edge A -> B means B depends on A.

KNOWN LIMITATIONS (a clean result is not proof of a clean pipeline):
  * Code is treated as straight-line: branches are visited in lexical order, not modelled.
  * Top-level functions and methods of top-level classes are analysed only where they are CALLED
    (arguments bound to parameters, up to MAX_INLINE_DEPTH). Unreferenced code is invisible.
  * Only class names listed in categories.py are tracked. Hand-rolled statistics are caught only by the
    .mean()/.std()/... heuristic, and only when the value flows into a split.
  * A Pipeline object has no split node; analyze_pipeline ASSUMES pipeline.fit() receives training data only.
"""

from __future__ import annotations

import ast
from functools import partial
from pathlib import Path

import networkx as nx

from .categories import CV_FUNCS, CV_SPLITTERS, LTYPE, SEARCH_CV, SEVERITY, category_of
from .schema import Finding, LayerResult

LAYER = 1
MAX_INLINE_DEPTH = 8
FIT_KINDS = {"fit", "fit_transform", "fit_resample"}
FIT_OR_STAT = FIT_KINDS | {"stat"}
CAT_FIT = ("preproc", "selection", "target_encoding", "oversample")
STAT_FUNCS = {"mean", "std", "median", "var", "min", "max", "quantile"}
_UNKNOWN_FIT = {
    "fit_transform": ("preproc", "<unknown transformer>"),
    "fit_resample": ("oversample", "<unknown sampler>"),
}


def _fit_category(method, cls):
    """(category, class label) of a fit-like call; unknown classes are classified by method name."""
    cat = category_of(cls)
    if cat or method not in _UNKNOWN_FIT:
        return cat, cls
    return _UNKNOWN_FIT[method][0], cls or _UNKNOWN_FIT[method][1]


# --------------------------------------------------------------------------- graph building
class _Builder(ast.NodeVisitor):
    def __init__(self, tree):
        self.g = nx.DiGraph()
        self.cur, self.gcur = {}, None  # current scope; enclosing scope while a function is inlined
        self.objcls = {}  # variable -> class name of the object last assigned to it
        self.funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.methods = {
            (c.name, m.name): m
            for c in tree.body
            if isinstance(c, ast.ClassDef)
            for m in c.body
            if isinstance(m, ast.FunctionDef)
        }
        self.ret_stack = []  # one set of returned sources per active inline; its length is the inline depth
        self.loop_depth = 0  # >0 while inside a CV loop body
        self.n = 0

    def _new(self, **attrs):
        nid = f"n{self.n}"
        self.n += 1
        self.g.add_node(nid, **attrs)
        return nid

    def _node(self, kind, srcs, lineno, **attrs):
        """Add an operation node fed by `srcs`; returns {node} so callers can chain it as a source."""
        nid = self._new(kind=kind, lineno=lineno, **attrs)
        self.g.add_edges_from((s, nid) for s in srcs)
        return {nid}

    def _read(self, name):
        if name in self.cur:
            return self.cur[name]
        if self.gcur is not None and name in self.gcur:
            return self.gcur[name]
        self.cur[name] = self._new(kind="var", name=name, lineno=0)
        return self.cur[name]

    def _assign(self, name, srcs, lineno):
        (nid,) = self._node("var", srcs, lineno, name=name)
        self.cur[name] = nid
        return nid

    @staticmethod
    def _fname(f):
        return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None

    def _recv_class(self, recv):
        if isinstance(recv, ast.Name):
            return self.objcls.get(recv.id)
        if isinstance(recv, ast.Call):
            name = self._fname(recv.func)
            return "Pipeline" if name == "make_pipeline" else name
        return None

    # ---- expressions: return the graph nodes the expression's value depends on
    def _expr(self, node):
        if isinstance(node, ast.Name):
            return {self._read(node.id)} if isinstance(node.ctx, ast.Load) else set()
        if isinstance(node, ast.Call):
            return self._call(node)
        return set().union(*(self._expr(ch) for ch in ast.iter_child_nodes(node)))

    def _call(self, node):
        fname, ln = self._fname(node.func), node.lineno
        arg_list = [self._expr(a) for a in node.args]
        kw_map = {k.arg: self._expr(k.value) for k in node.keywords}
        args = set().union(*arg_list, *kw_map.values())
        if isinstance(node.func, ast.Name) and fname in self.funcs:
            return self._inline(self.funcs[fname], arg_list, kw_map, ln)
        recv = node.func.value if isinstance(node.func, ast.Attribute) else None
        recv_srcs = self._expr(recv) if recv is not None else set()
        cls = self._recv_class(recv) if recv is not None else None
        if (cls, fname) in self.methods:
            return self._inline(self.methods[cls, fname], arg_list, kw_map, ln, bound=recv_srcs)

        if fname == "train_test_split":
            return self._node("split", args, ln, name=fname)
        if fname in CV_FUNCS:
            return self._node("cv_eval", args, ln, name=fname)
        if recv is not None and fname == "split" and cls in CV_SPLITTERS:
            return self._node("cv_split", args, ln, name=f"{cls}.split")
        if recv is not None and fname == "fit" and cls in SEARCH_CV:
            return self._node("cv_eval", args, ln, name=f"{cls}.fit")
        if recv is not None and fname in FIT_KINDS and (out := self._fit_call(node, fname, cls, recv, args)):
            return out
        if fname in STAT_FUNCS:
            return self._node(
                "stat",
                args | recv_srcs,
                ln,
                name=f"pandas/numpy .{fname}()",
                cls=f".{fname}()",
                category="preproc",
                flow_only=True,
            )
        return args | recv_srcs

    def _fit_call(self, node, fname, cls, recv, args):
        """Node for a fit/fit_transform/fit_resample call on a recognised estimator, else None."""
        cat, cls = _fit_category(fname, cls)
        if not cat:
            return None
        supervised = len(node.args) >= 2 or any(k.arg == "y" for k in node.keywords)
        out = self._node(
            fname,
            args,
            node.lineno,
            name=f"{cls}.{fname}",
            cls=cls,
            category=cat,
            in_cv_loop=self.loop_depth > 0,
            supervised=supervised,
        )
        if isinstance(recv, ast.Name):  # the object's state now depends on this fit
            self._assign(recv.id, out, node.lineno)
        return out

    def _inline(self, fdef, arg_list, kw_map, ln, bound=None):
        """Analyse a local function/method body at its call site, binding parameters to the arguments."""
        if len(self.ret_stack) >= MAX_INLINE_DEPTH:
            return set().union(*arg_list, *kw_map.values(), bound or set())
        positional = ([bound] if bound is not None else []) + arg_list
        saved, saved_g = self.cur, self.gcur
        self.gcur = saved if saved_g is None else {**saved_g, **saved}
        self.cur = {}
        for i, param in enumerate(a.arg for a in fdef.args.args):
            self._assign(param, positional[i] if i < len(positional) else kw_map.get(param, set()), ln)
        self.ret_stack.append(set())
        for st in fdef.body:
            self.visit(st)
        self.cur, self.gcur = saved, saved_g
        return self.ret_stack.pop()

    # ---- statements
    def _targets(self, t):
        if isinstance(t, ast.Name):
            return [t.id]
        if isinstance(t, (ast.Tuple, ast.List)):
            return [n for e in t.elts for n in self._targets(e)]
        if isinstance(t, (ast.Subscript, ast.Attribute)):
            while isinstance(t, (ast.Subscript, ast.Attribute)):
                t = t.value
            return [t.id] if isinstance(t, ast.Name) else []
        return []

    def visit_Assign(self, node):
        srcs = self._expr(node.value)
        call = node.value if isinstance(node.value, ast.Call) else None
        cls = self._recv_class(call) if call else None
        is_split = call is not None and self._fname(call.func) == "train_test_split"
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.objcls[t.id] = cls  # also clears a stale class when the name is rebound
            if is_split and isinstance(t, (ast.Tuple, ast.List)):
                for idx, e in enumerate(t.elts):  # outputs alternate train, test, train, test, ...
                    if isinstance(e, ast.Name):
                        nid = self._assign(e.id, srcs, node.lineno)
                        self.g.nodes[nid]["split_part"] = "test" if idx % 2 else "train"
                continue
            for name in self._targets(t):
                # X[col] = ... / X.attr = ... mutates the existing variable
                is_item = isinstance(t, (ast.Subscript, ast.Attribute))
                prior = {self.cur[name]} if is_item and name in self.cur else set()
                self._assign(name, srcs | prior, node.lineno)

    def visit_AugAssign(self, node):
        for name in self._targets(node.target):
            self._assign(name, self._expr(node.value) | {self._read(name)}, node.lineno)

    def visit_Expr(self, node):
        self._expr(node.value)

    def visit_Return(self, node):
        srcs = self._expr(node.value) if node.value else set()
        if self.ret_stack:
            self.ret_stack[-1] |= srcs

    def visit_FunctionDef(self, node):
        """Bodies are analysed only when called (see _inline)."""

    visit_AsyncFunctionDef = visit_ClassDef = visit_FunctionDef

    def visit_For(self, node):
        srcs = self._expr(node.iter)
        in_cv = any(self.g.nodes[s]["kind"] == "cv_split" for s in srcs)
        for name in self._targets(node.target):
            self._assign(name, srcs, node.lineno)
        self.loop_depth += in_cv
        for st in node.body:
            self.visit(st)
        self.loop_depth -= in_cv
        for st in node.orelse:
            self.visit(st)


# --------------------------------------------------------------------------- rules
def _nodes(g, *kinds):
    return [n for n, d in g.nodes(data=True) if d["kind"] in kinds]


def _fits(g, cats, kinds=FIT_KINDS):
    return [n for n, d in g.nodes(data=True) if d["kind"] in kinds and d.get("category") in cats]


def _var_inputs(g, n):
    return {p for p in g.predecessors(n) if g.nodes[p]["kind"] == "var"}


def _step_label(d):
    return f"{d['cls']}@L{d['lineno']}" if d["lineno"] else d["cls"]


def _at(d):
    return f"line {d['lineno']}" if d["lineno"] else "this step"


def _before_split_message(d, where):
    if d["kind"] == "stat":
        return 3, (
            f"{d['name']} at {_at(d)} computes a full-data statistic that flows into data split later "
            f"({where}); held-out rows influence the transformed values. (Heuristic: detected from the "
            f"data flow, not from a fitted estimator object.)"
        )
    cat = d["category"]
    msg = (
        f"{d['name']}() at {_at(d)} is fit on data that is split later ({where}); "
        f"its learned statistics include held-out rows."
    )
    if d.get("supervised"):
        msg += " It is supervised (receives y), so target information is baked into the transform."
    elif cat == "preproc":
        msg += (
            " (For plain scalers/imputers the practical inflation is usually small, "
            "but the evaluation is still invalid.)"
        )
    return SEVERITY[cat] + (d.get("supervised") and cat == "preproc"), msg


def _rule_before_split(g, cats, kinds=FIT_KINDS):
    """Flag fits whose data is split afterwards (the fit does not descend from a split)."""
    split_ancestors = {S: nx.ancestors(g, S) for S in _nodes(g, "split")}
    out = []
    for F in _fits(g, cats, kinds):
        d = g.nodes[F]
        if any(g.nodes[a]["kind"] in ("split", "cv_split") for a in nx.ancestors(g, F)):
            continue  # fit happens on already-split data
        inputs = _var_inputs(g, F)
        # statistic nodes must flow into the split; fitted estimators also count when they share its input data
        shares_input = not d.get("flow_only")
        hit = next((S for S, anc in split_ancestors.items() if F in anc or (shares_input and inputs & anc)), None)
        if hit is None:
            continue
        s = g.nodes[hit]
        where = _at(s) if s["lineno"] else f"step '{s['name']}'"
        sev, msg = _before_split_message(d, where)
        out.append(
            Finding(
                LAYER,
                _step_label(d),
                LTYPE[d["category"]],
                sev,
                msg,
                "Split first, fit on the training part only, then transform validation/test "
                "(or put the step inside an sklearn Pipeline and fit the Pipeline on train).",
            )
        )
    return out


def rule_fit_on_test(g):
    """A learning step whose input descends from the TEST output of train_test_split."""
    return [
        Finding(
            LAYER,
            _step_label(d),
            "fit-on-test-data",
            4,
            f"{d['name']}() at line {d['lineno']} is fit on data derived from the test partition of "
            f"train_test_split; the model/preprocessor learns from held-out rows.",
            "Fit only on the training partition; use .transform() on the test partition.",
        )
        for F in _fits(g, CAT_FIT)
        for d in [g.nodes[F]]
        if any(g.nodes[a].get("split_part") == "test" for v in _var_inputs(g, F) for a in nx.ancestors(g, v) | {v})
    ]


def rule_cv_leakage(g):
    fits = _fits(g, CAT_FIT)
    out = []
    for C in _nodes(g, "cv_split", "cv_eval"):
        c, anc = g.nodes[C], nx.ancestors(g, C)
        out += [
            Finding(
                LAYER,
                _step_label(d),
                "cv-preprocessing-leakage",
                SEVERITY[d["category"]],
                f"{d['name']}() at line {d['lineno']} is fit once on data that feeds the cross-validation "
                f"at line {c['lineno']} ({c['name']}); every validation fold influenced the fitted state.",
                "Fit the transform inside each fold: wrap it in a Pipeline and pass that to the CV call.",
            )
            for F in fits
            if F in anc
            for d in [g.nodes[F]]
        ]
    # a fit inside a CV loop must descend from the loop's cv_split node (i.e. use the fold's indices)
    fold_data = set().union(*(nx.descendants(g, c) for c in _nodes(g, "cv_split")))
    out += [
        Finding(
            LAYER,
            _step_label(d),
            "cv-loop-leakage",
            SEVERITY[d["category"]],
            f"{d['name']}() at line {d['lineno']} runs inside a CV loop but its input does not depend on "
            f"the fold's train indices, so it is fit on the full data each iteration.",
            "Index the data with the fold's train indices before fitting.",
        )
        for F in fits
        for d in [g.nodes[F]]
        if d.get("in_cv_loop") and not _var_inputs(g, F) & fold_data
    ]
    return out


RULES = [
    partial(_rule_before_split, cats=("preproc",), kinds=FIT_OR_STAT),
    rule_fit_on_test,
    *(partial(_rule_before_split, cats=(cat,)) for cat in ("selection", "target_encoding", "oversample")),
    rule_cv_leakage,
]


def run_rules(g):
    return [f for rule in RULES for f in rule(g)]


# --------------------------------------------------------------------------- entry points
def build_graph_from_source(source: str):
    tree = ast.parse(source)
    builder = _Builder(tree)
    builder.visit(tree)
    return builder.g


def analyze_source(source: str, filename: str = "<source>") -> LayerResult:
    try:
        g = build_graph_from_source(source)
    except SyntaxError as e:
        return LayerResult.skipped(LAYER, f"{filename}: cannot parse ({e})")
    if not _nodes(g, "split", "cv_split", "cv_eval"):
        return LayerResult.skipped(
            LAYER,
            f"{filename}: no train_test_split / cross-validation call reached from "
            f"top-level code; nothing to order fits against.",
        )
    notes = [
        "Source analysis is straight-line: branches are not modelled and functions/methods are analysed only "
        "where called; pandas-level statistics are caught only by the .mean()/.std() flow heuristic."
    ]
    return LayerResult(LAYER, "ok", findings=run_rules(g), notes=notes)


def analyze_file(path: str) -> LayerResult:
    return analyze_source(Path(path).read_text(), path)


_STEP_KIND = {"select": "fit_transform", "target_encode": "fit_transform", "oversample": "fit_resample"}
_STEP_CAT = {
    "fit": "preproc",
    "fit_transform": "preproc",
    "select": "selection",
    "target_encode": "target_encoding",
    "oversample": "oversample",
}
_STEP_OPS = {"split", "transform"} | set(_STEP_CAT)
_OP_FOR_CAT = {"selection": "select", "target_encoding": "target_encode", "oversample": "oversample"}


def analyze_steps(steps) -> LayerResult:
    """steps: [(name, callable, op_type)] with op_type in split|fit|fit_transform|transform|select|target_encode|
    oversample. Steps form a chain; a fit step with no preceding split step is flagged."""
    g = nx.DiGraph()
    for i, (name, _fn, op) in enumerate(steps):
        if op not in _STEP_OPS:
            raise ValueError(f"unknown op_type {op!r} for step {name!r}")
        attrs = {"kind": _STEP_KIND.get(op, op), "name": name, "cls": name, "lineno": 0}
        if op in _STEP_CAT:
            attrs.update(category=_STEP_CAT[op], supervised=op in ("select", "target_encode"))
        g.add_node(i, **attrs)
        if i:
            g.add_edge(i - 1, i)
    if not _nodes(g, "split"):
        return LayerResult.skipped(LAYER, "no 'split' step declared; cannot order fits against a split.")
    return LayerResult(LAYER, "ok", findings=run_rules(g))


def analyze_pipeline(pipeline) -> LayerResult:
    """Lower an sklearn Pipeline/ColumnTransformer to steps behind a prepended split step. A Pipeline object has no
    split of its own, so this ASSUMES pipeline.fit() is given training data only."""
    from sklearn.compose import ColumnTransformer  # lazy: source analysis needs no sklearn
    from sklearn.pipeline import Pipeline

    steps = [("train_test_split", None, "split")]

    def walk(obj):
        if isinstance(obj, Pipeline):
            for _name, step in obj.steps:
                walk(step)
        elif isinstance(obj, ColumnTransformer):
            for _name, step, _cols in obj.transformers:
                if not isinstance(step, str):  # "drop" / "passthrough"
                    walk(step)
        else:
            cls = type(obj).__name__
            steps.append((cls, obj, _OP_FOR_CAT.get(category_of(cls), "fit_transform")))

    walk(pipeline)
    res = analyze_steps(steps)
    res.notes.append("Assumed pipeline.fit() receives training data only; a Pipeline object cannot prove it.")
    return res
