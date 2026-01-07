"""
MPC Hyperparameter Tuning Script

This script runs across many data files to find optimal MPC parameters,
then saves them to a JSON file that the MPC controller loads at runtime.

Usage:
    python tune_mpc.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_files 1000
"""

import numpy as np
import json
import os
from pathlib import Path
from functools import partial
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


def evaluate_single_file(args):
    """Evaluate a single file with given parameters. Run in subprocess."""
    data_file, params, model_path = args
    
    # Import here to avoid multiprocessing issues
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    from controllers.mpc import Controller
    
    try:
        model = TinyPhysicsModel(model_path, debug=False)
        controller = Controller(params_dict=params)
        sim = TinyPhysicsSimulator(model, str(data_file), controller=controller, debug=False)
        cost = sim.rollout()
        return cost
    except Exception as e:
        return {'lataccel_cost': 100, 'jerk_cost': 100, 'total_cost': 5100}


def evaluate_params(params, data_files, model_path, num_workers=8):
    """Evaluate parameters across multiple files in parallel."""
    args_list = [(f, params, model_path) for f in data_files]
    
    costs = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(evaluate_single_file, args) for args in args_list]
        for future in futures:
            try:
                costs.append(future.result(timeout=30))
            except:
                costs.append({'lataccel_cost': 100, 'jerk_cost': 100, 'total_cost': 5100})
    
    avg_lataccel = np.mean([c['lataccel_cost'] for c in costs])
    avg_jerk = np.mean([c['jerk_cost'] for c in costs])
    avg_total = np.mean([c['total_cost'] for c in costs])
    
    return {'lataccel_cost': avg_lataccel, 'jerk_cost': avg_jerk, 'total_cost': avg_total}


def get_default_params():
    """Get default parameter values (good starting point)."""
    return {
        'kp': 0.3,
        'ki': 0.05,
        'kd': 0.1,
        'ff_target': 0.2,
        'ff_future': 0.05,
        'ff_roll': 0.1,
        'action_rate_limit': 0.4,
        'lookahead_steps': 5,
        'error_filter_alpha': 0.3,
    }


def get_param_ranges():
    """Define search ranges for each parameter."""
    return {
        'kp': (0.1, 0.5),
        'ki': (0.01, 0.15),
        'kd': (0.02, 0.25),
        'ff_target': (0.1, 0.4),
        'ff_future': (0.01, 0.15),
        'ff_roll': (0.05, 0.25),
        'action_rate_limit': (0.2, 0.6),
        'lookahead_steps': (3, 10),
        'error_filter_alpha': (0.1, 0.5),
    }


def sample_params(ranges):
    """Sample random parameters from ranges."""
    params = {}
    for key, (low, high) in ranges.items():
        if key == 'lookahead_steps':
            params[key] = int(np.random.randint(low, high + 1))
        else:
            params[key] = np.random.uniform(low, high)
    return params


def mutate_params(params, ranges, mutation_rate=0.3):
    """Mutate parameters slightly."""
    new_params = params.copy()
    for key, (low, high) in ranges.items():
        if np.random.random() < mutation_rate:
            if key == 'lookahead_steps':
                delta = np.random.randint(-2, 3)
                new_params[key] = int(np.clip(params[key] + delta, low, high))
            else:
                delta = (high - low) * np.random.uniform(-0.2, 0.2)
                new_params[key] = np.clip(params[key] + delta, low, high)
    return new_params


def evolutionary_search(data_files, model_path, n_generations=20, population_size=15, 
                        elite_size=3, files_per_eval=50, num_workers=8):
    """
    Evolutionary search for optimal parameters.
    """
    ranges = get_param_ranges()
    
    # Initialize population
    population = [get_default_params()]  # Start with defaults
    for _ in range(population_size - 1):
        population.append(sample_params(ranges))
    
    best_ever_params = None
    best_ever_cost = float('inf')
    
    print(f"Starting evolutionary search: {n_generations} generations, population={population_size}")
    print(f"Evaluating on {files_per_eval} files per candidate\n")
    
    for gen in range(n_generations):
        print(f"Generation {gen + 1}/{n_generations}")
        
        # Evaluate all candidates
        eval_files = list(np.random.choice(data_files, size=min(files_per_eval, len(data_files)), replace=False))
        
        scores = []
        for i, params in enumerate(tqdm(population, desc="  Evaluating")):
            result = evaluate_params(params, eval_files, model_path, num_workers)
            scores.append((result['total_cost'], params, result))
        
        # Sort by cost
        scores.sort(key=lambda x: x[0])
        
        # Report best
        best_cost, best_params, best_result = scores[0]
        print(f"  Best: total={best_cost:.2f} (lataccel={best_result['lataccel_cost']:.3f}, jerk={best_result['jerk_cost']:.3f})")
        
        if best_cost < best_ever_cost:
            best_ever_cost = best_cost
            best_ever_params = best_params.copy()
            print(f"  *** New best ever! ***")
        
        # Create next generation
        # Keep elite
        next_population = [s[1] for s in scores[:elite_size]]
        
        # Breed from top half
        top_half = [s[1] for s in scores[:population_size // 2]]
        while len(next_population) < population_size:
            parent = np.random.choice(len(top_half))
            child = mutate_params(top_half[parent], ranges)
            next_population.append(child)
        
        population = next_population
        print()
    
    return best_ever_params, best_ever_cost


def local_refine(params, data_files, model_path, files_per_eval=100, num_workers=8):
    """
    Local refinement around best parameters.
    """
    print("Local refinement phase...")
    ranges = get_param_ranges()
    
    eval_files = data_files[:files_per_eval]
    current_params = params.copy()
    current_result = evaluate_params(current_params, eval_files, model_path, num_workers)
    current_cost = current_result['total_cost']
    
    print(f"  Starting cost: {current_cost:.2f}")
    
    # Try small adjustments to each parameter
    for key in params:
        if key == 'lookahead_steps':
            deltas = [-1, 1]
        else:
            base = params[key]
            deltas = [-0.1 * base, -0.05 * base, 0.05 * base, 0.1 * base]
        
        for delta in deltas:
            test_params = current_params.copy()
            if key == 'lookahead_steps':
                test_params[key] = int(np.clip(current_params[key] + delta, *ranges[key]))
            else:
                test_params[key] = np.clip(current_params[key] + delta, *ranges[key])
            
            result = evaluate_params(test_params, eval_files, model_path, num_workers)
            
            if result['total_cost'] < current_cost:
                current_cost = result['total_cost']
                current_params = test_params.copy()
                print(f"  Improved {key}: {current_cost:.2f}")
    
    print(f"  Final refined cost: {current_cost:.2f}")
    return current_params, current_cost


def main():
    parser = argparse.ArgumentParser(description='Tune MPC hyperparameters across many files')
    parser.add_argument('--model_path', type=str, default='./models/tinyphysics.onnx')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--num_files', type=int, default=1000, help='Number of files to use for tuning')
    parser.add_argument('--generations', type=int, default=15, help='Number of evolution generations')
    parser.add_argument('--population', type=int, default=12, help='Population size')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--output', type=str, default='./models/mpc_params.json')
    args = parser.parse_args()
    
    # Get data files
    data_path = Path(args.data_path)
    data_files = sorted(data_path.glob('*.csv'))[:args.num_files]
    
    print(f"="*60)
    print(f"MPC Parameter Tuning")
    print(f"="*60)
    print(f"Data files: {len(data_files)}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}")
    print()
    
    # Run evolutionary search
    best_params, best_cost = evolutionary_search(
        data_files, args.model_path,
        n_generations=args.generations,
        population_size=args.population,
        files_per_eval=min(50, len(data_files)),
        num_workers=args.workers
    )
    
    print(f"\nBest from evolution: {best_cost:.2f}")
    
    # Local refinement
    refined_params, refined_cost = local_refine(
        best_params, data_files, args.model_path,
        files_per_eval=min(100, len(data_files)),
        num_workers=args.workers
    )
    
    # Final evaluation on all files
    print(f"\nFinal evaluation on {len(data_files)} files...")
    final_result = evaluate_params(refined_params, data_files, args.model_path, args.workers)
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Lataccel cost: {final_result['lataccel_cost']:.4f}")
    print(f"Jerk cost:     {final_result['jerk_cost']:.4f}")
    print(f"Total cost:    {final_result['total_cost']:.4f}")
    
    print(f"\nOptimal parameters:")
    for k, v in refined_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # Save to file
    output_data = {
        'params': refined_params,
        'results': {
            'lataccel_cost': float(final_result['lataccel_cost']),
            'jerk_cost': float(final_result['jerk_cost']),
            'total_cost': float(final_result['total_cost']),
        },
        'num_files_evaluated': len(data_files),
    }
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nParameters saved to: {args.output}")
    print("The MPC controller will automatically load these parameters.")


if __name__ == '__main__':
    main()
