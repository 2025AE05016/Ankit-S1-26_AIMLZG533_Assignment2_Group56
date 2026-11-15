"""Optional resumable command-line entry point; use notebooks for embedded submission outputs."""
import argparse
from udl.pipeline import load_config, run_part_a, run_part_b
from udl.reporting import build_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['a', 'b', 'all', 'report'])
    parser.add_argument('--smoke', action='store_true', help='Tiny pipeline check, never submission results')
    parser.add_argument('--final', action='store_true', help='Require full results, discussion and lab screenshot')
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args(); cfg = load_config(args.config, smoke=args.smoke, require_training_device=args.part != 'report')
    if args.part in ('a', 'all'): run_part_a(cfg)
    if args.part in ('b', 'all'): run_part_b(cfg)
    if args.part in ('report', 'all'): build_report(cfg, final=args.final)

if __name__ == '__main__': main()
