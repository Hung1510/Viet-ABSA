"""Tiny config loader: a YAML file supplies argparse defaults, so `--config
config/default.yaml` replaces a long CLI invocation, but any CLI flag you DO
pass still overrides the file (argparse default-override semantics)."""
import yaml


def apply_config_defaults(parser, args_so_far):
    """Call after `ap.add_argument("--config", ...)` and a first partial
    parse_known_args(); loads the YAML (if given) and calls set_defaults so
    the real parse picks up file values, with CLI flags still taking
    precedence over them.
    """
    if not args_so_far.config:
        return
    with open(args_so_far.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    flat = {
        "encoder": cfg.get("encoder"),
        "max_len": cfg.get("max_len"),
        "batch": cfg.get("batch"),
        "epochs": cfg.get("epochs"),
        "lr": cfg.get("lr"),
        "warmup": cfg.get("warmup"),
        "seed": cfg.get("seed"),
        "patience": cfg.get("patience"),
        "augment_p": cfg.get("augment_p"),
    }
    paths = cfg.get("paths", {})
    if "data_dir" in paths:
        flat["data_dir"] = paths["data_dir"]
    flat = {k: v for k, v in flat.items() if v is not None}
    known_dests = {action.dest for action in parser._actions}
    flat = {k: v for k, v in flat.items() if k in known_dests}
    parser.set_defaults(**flat)
