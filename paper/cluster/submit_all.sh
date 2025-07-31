for script in src/sims/*.py; do
    sbatch --export=SCRIPT="$script" cluster/run_python.run
done