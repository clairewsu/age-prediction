SEED=$1

if [ -z "$SEED" ]; then
  echo "Usage: bash run_permutation.sh <seed>"
  exit 1
fi

FILES=(
  "adni.cdr0.csv"
  "adni.cdr.5.csv"
  "adni.cdr1.csv"
  "adni.cdr2and3.csv"
)

for f in "${FILES[@]}"
do
  echo "Running permutation on $f"

  python permutation.py \
    --csv "$f" \
    --seed "$SEED" \
    --feature-stats ../feature_stats_different.csv \
    --out "${f%.csv}_perm.csv"

done
