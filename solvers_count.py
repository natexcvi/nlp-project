import re
from collections import defaultdict

# Path to the YAML file
file_path = '/Users/ariellearabov/Documents/TAU_CS/master/NLP/nlp-project/reports/experiment_report_20250504_220711.yaml'

# Only consider these solvers
allowed_solvers = {
    "ChainOfThoughtSolver",
    "DynamicFewShotSolver",
    "SelfRefineSolver"
}

# Initialize data structures
problem_scores = defaultdict(dict)
solvers = set()

# Read and manually parse the YAML file
with open(file_path, 'r') as f:
    current_solver = None
    current_problem = None
    for line in f:
        # Detect solver entries at indent level 2
        match_solver = re.match(r'^  (\w+):$', line)
        if match_solver:
            solver_name = match_solver.group(1)
            if solver_name in allowed_solvers:
                current_solver = solver_name
                solvers.add(current_solver)
            else:
                current_solver = None
            continue

        # Detect problem entries at indent level 4 under a valid solver
        if current_solver:
            match_problem = re.match(r'^\s{4}([^:]+):$', line)
            if match_problem:
                current_problem = match_problem.group(1).strip()
                continue

            # Detect avg_score lines under a valid solver and problem
            if current_problem and 'avg_score:' in line:
                score = float(line.split(':')[1].strip())
                problem_scores[current_problem][current_solver] = score

# Ensure every allowed solver has an entry for each problem (default 0.0)
for problem in problem_scores:
    for solver in allowed_solvers:
        problem_scores[problem].setdefault(solver, 0.0)

# Calculate counts and lists of uniquely solved problems
only_solved_counts = {solver: 0 for solver in allowed_solvers}
only_solved_problems = {solver: [] for solver in allowed_solvers}
strictly_better_counts = {solver: 0 for solver in allowed_solvers}

for problem, scores in problem_scores.items():
    # Only solved by one solver (others = 0)
    for solver in allowed_solvers:
        if scores.get(solver, 0) > 0 and all(scores.get(other, 0) == 0 for other in allowed_solvers if other != solver):
            only_solved_counts[solver] += 1
            only_solved_problems[solver].append(problem)
    # Strictly better than all others
    for solver in allowed_solvers:
        if all(scores.get(solver, 0) > scores.get(other, 0) for other in allowed_solvers if other != solver):
            strictly_better_counts[solver] += 1

# Display results
print("Problems only solved by each solver:")
for solver, count in only_solved_counts.items():
    print(f"\n  {solver}: {count}")
    for prob in only_solved_problems[solver]:
        print(f"    - {prob}")

print("\nProblems each solver solved strictly better than all others:")
for solver, count in strictly_better_counts.items():
    print(f"  {solver}: {count}")
