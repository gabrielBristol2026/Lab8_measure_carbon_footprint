#!/bin/bash

# =============================================================
# SOLUTION: run_experiments.sh
# Advanced Automation for ML Experiments
# =============================================================

function BREAKPOINT() {
  BREAKPOINT_NAME=$1
  echo "Enter breakpoint $BREAKPOINT_NAME"
  set +e
  /bin/bash
  BREAKPOINT_EXIT_CODE=$?
  set -e
  if [[ $BREAKPOINT_EXIT_CODE -eq 0 ]]; then
    echo "Continue after breakpoint $BREAKPOINT_NAME"
  else
    echo "Terminate after breakpoint $BREAKPOINT_NAME"
    exit $BREAKPOINT_EXIT_CODE
  fi
}


# --- Task 1: Dynamic Hyperparameters ---
DEFAULT_LRS="0.01,0.001,0.0001"
DEFAULT_EPOCHS=2
DEFAULT_MODEL="cnn"

LRS_STRING=${1:-$DEFAULT_LRS}
EPOCHS=${2:-$DEFAULT_EPOCHS}
MODEL=${3:-$DEFAULT_MODEL}

case "$MODEL" in
  cnn|vgg|vit)
    ;;
  *)
    echo "ERROR: Unsupported model '$MODEL'. Choose one of: cnn, vgg, vit"
    exit 1
    ;;
esac

IFS="," read -ra LEARNING_RATES <<< "$LRS_STRING"

echo "Learning rates to test: ${LEARNING_RATES[*]}"
echo "Number of epochs: $EPOCHS"
echo "Selected model: $MODEL"

# --- Task 2: Timestamped Output Directory ---
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
MAIN_LOG_DIR="experiment_${MODEL}_${TIMESTAMP}"
mkdir -p "$MAIN_LOG_DIR"

echo "Starting experiment. All results will be saved in: $MAIN_LOG_DIR"
echo ""

# --- Task 3: Prepare Summary File ---
SUMMARY_FILE="$MAIN_LOG_DIR/summary.csv"
echo "model,learning_rate,final_accuracy" > "$SUMMARY_FILE"

# =============================================================
# Main Training Loop
# =============================================================
for lr in "${LEARNING_RATES[@]}"
do
  echo "-------------------------------------------------"
  echo "---- STARTING TRAINING - MODEL: $MODEL, LR: $lr, Epochs: $EPOCHS ----"
  echo "-------------------------------------------------"

  python3 solution_train_cnn_V2.py --model "$MODEL" --lr "$lr" --epochs "$EPOCHS" --log-dir "$MAIN_LOG_DIR"

  if [ $? -ne 0 ]; then
    echo "ERROR: Training failed for model $MODEL with learning rate $lr. Exiting."
    exit 1
  fi

  CSV_LOG="$MAIN_LOG_DIR/log_lr_${lr}.csv"
  FINAL_ACCURACY=$(tail -n 1 "$CSV_LOG" | awk -F',' '{print $3}')

  echo "$MODEL,$lr,$FINAL_ACCURACY" >> "$SUMMARY_FILE"

  echo "-------------------------------------------------"
  echo "---- FINISHED TRAINING - MODEL: $MODEL, LEARNING RATE: $lr ----"
  echo "-------------------------------------------------"
  echo ""
done

# =============================================================
# Task 4 (Bonus): Automated Plotting
# =============================================================
echo "Generating result plots..."
python3 solution_plot_results.py --dir "$MAIN_LOG_DIR"

echo ""
echo "============================================="
echo "All training experiments have been completed."
echo "Results are in the '$MAIN_LOG_DIR' directory."
echo "  - CSV logs:    log_lr_*.csv"
echo "  - Summary:     summary.csv"
echo "  - Plots:       loss_comparison.png, accuracy_comparison.png"
echo "============================================="
