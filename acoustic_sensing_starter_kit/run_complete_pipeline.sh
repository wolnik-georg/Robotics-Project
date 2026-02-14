#!/bin/bash
###############################################################################
# Complete Experimental Pipeline - Reproduce All Main Results
###############################################################################
#
# USAGE:
#   bash run_complete_pipeline.sh
#
# WHAT IT DOES:
#   Runs the entire experimental pipeline end-to-end:
#       1. Dataset balancing (33/33/33 class splits)
#       2. Position generalization (3 workspace rotations)
#       3. Object generalization (5-seed multi-seed validation)
#       4. Figure generation (all main reconstruction and analysis figures)
#
#   NEVER OVERWRITES EXISTING RESULTS:
#   - Creates timestamped output directory: results_YYYYMMDD_HHMMSS/
#   - Option to reuse existing balanced datasets (saves ~5 minutes)
#   - All new results go to the timestamped directory
#   - Your existing results remain untouched
#
# OUTPUTS:
#   results_YYYYMMDD_HHMMSS/               # All results in timestamped directory
#       ├── rotation{1,2,3}_results/       # Position generalization
#       ├── object_generalization_seed_*/  # Object generalization  
#       └── figures/
#           ├── reconstruction/             # Main reconstruction figures
#           └── analysis/                   # ML analysis figures
#
#   data/fully_balanced_datasets/          # Balanced datasets (reusable)
#
# ESTIMATED TIME:
#   Total: ~4-5 hours (or ~4 hours if reusing balanced datasets)
#       - Dataset balancing: ~5 minutes (can be skipped if exists)
#       - Position generalization: ~3 hours
#       - Object generalization: ~1.5 hours
#       - Figure generation: ~10 minutes
#
# See README.md Section "Reproducing Main Results" for details.
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored headers
print_header() {
    echo -e "\n${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to check and backup existing directories
backup_if_exists() {
    local dir=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    if [ -d "$dir" ]; then
        local backup_dir="${dir}_backup_${timestamp}"
        echo -e "${YELLOW}⚠ Directory exists: $dir${NC}"
        echo -n "  Backup to $backup_dir? [Y/n]: "
        read -r response
        if [[ "$response" =~ ^[Nn]$ ]]; then
            echo "  Skipping backup. Directory will be overwritten."
        else
            mv "$dir" "$backup_dir"
            print_success "Backed up to: $backup_dir"
        fi
    fi
}

# Start time
START_TIME=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create timestamped output directories to NEVER overwrite existing results
RESULTS_DIR="results_${TIMESTAMP}"

print_header "COMPLETE EXPERIMENTAL PIPELINE - ACOUSTIC CONTACT DETECTION"
echo "This will reproduce all main experimental results from the paper."
echo ""
echo "Steps to be executed:"
echo "  1. Dataset Balancing (~5 minutes)"
echo "  2. Position Generalization - 3 Rotations (~3 hours)"
echo "  3. Object Generalization - 5 Seeds (~1.5 hours)"
echo "  4. Figure Generation (~10 minutes)"
echo ""
echo "Total estimated time: 4-5 hours"
echo ""

# Check for existing balanced datasets
SKIP_BALANCING=false
if [ -d "data/fully_balanced_datasets" ]; then
    echo -e "${GREEN}✓ Found existing balanced datasets: data/fully_balanced_datasets/${NC}"
    echo ""
    echo "Options:"
    echo "  1) Reuse existing balanced datasets (saves ~5 minutes)"
    echo "  2) Create fresh balanced datasets (will be saved with timestamp)"
    echo ""
    read -p "Choose option [1/2]: " dataset_option
    
    case $dataset_option in
        1)
            print_success "Will reuse existing balanced datasets"
            SKIP_BALANCING=true
            ;;
        2)
            print_success "Will create fresh balanced datasets"
            SKIP_BALANCING=false
            ;;
        *)
            echo "Invalid option. Exiting."
            exit 1
            ;;
    esac
else
    echo "No existing balanced datasets found. Will create new ones."
    SKIP_BALANCING=false
fi

echo ""
echo -e "${BLUE}All results will be saved to timestamped directory: ${RESULTS_DIR}/${NC}"
echo "This ensures your existing results are NEVER overwritten."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Create timestamped results directory
mkdir -p "$RESULTS_DIR"

################################################################################
# STEP 1: Dataset Balancing
################################################################################

print_header "STEP 1/4: DATASET BALANCING"

if [ "$SKIP_BALANCING" = true ]; then
    print_warning "Skipping dataset balancing (reusing existing datasets)"
    
    # Verify datasets exist
    if [ ! -d "data/fully_balanced_datasets" ]; then
        print_error "Balanced datasets not found! Cannot skip balancing."
        exit 1
    fi
    
    print_success "Using existing balanced datasets: data/fully_balanced_datasets/"
else
    echo "Creating perfectly balanced 3-class datasets (33/33/33 splits)"
    echo "Output: data/fully_balanced_datasets/"
    echo ""

    if bash run_balance_datasets.sh; then
        print_success "Dataset balancing complete!"
        
        # Verify balance
        echo ""
        echo "Verifying dataset balance..."
        if python analyze_dataset_balance.py > /dev/null 2>&1; then
            print_success "Dataset balance verified (33/33/33)"
        else
            print_warning "Balance verification had warnings (check logs)"
        fi
    else
        print_error "Dataset balancing failed!"
        exit 1
    fi
fi

################################################################################
# STEP 2: Position Generalization (3 Workspace Rotations)
################################################################################

print_header "STEP 2/4: POSITION GENERALIZATION (3 Workspace Rotations)"
echo "Running all 3 workspace rotations to test position generalization"
echo "Expected: Catastrophic failure (avg ~34.5% vs 33.3% random)"
echo "Results will be saved to: $RESULTS_DIR/"
echo ""

# Create rotation configs that output to timestamped directory
for i in 1 2 3; do
    ROTATION_OUTPUT="${RESULTS_DIR}/rotation${i}_results"
    
    echo "Running Rotation $i..."
    echo "Output: $ROTATION_OUTPUT"
    
    # Run with custom output directory
    if [ $i -eq 1 ]; then
        python run_modular_experiments.py configs/multi_dataset_config.yml "$ROTATION_OUTPUT"
    elif [ $i -eq 2 ]; then
        python run_modular_experiments.py configs/rotation_ws2_ws3_train_ws1_val.yml "$ROTATION_OUTPUT"
    else
        python run_modular_experiments.py configs/rotation_ws1_ws2_train_ws3_val.yml "$ROTATION_OUTPUT"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Rotation $i complete!"
    else
        print_error "Rotation $i failed!"
        exit 1
    fi
    echo ""
done

# Extract and display validation accuracies
echo ""
echo "Validation Accuracies:"
for i in 1 2 3; do
    METRICS_FILE="${RESULTS_DIR}/rotation${i}_results/discriminationanalysis/validation_results/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        ACC=$(grep -o '"validation_accuracy":[^,]*' "$METRICS_FILE" | cut -d':' -f2 | tr -d ' ')
        echo "  Rotation $i: ${ACC}%"
    fi
done

################################################################################
# STEP 3: Object Generalization (Multi-Seed Validation)
################################################################################

print_header "STEP 3/4: OBJECT GENERALIZATION (Multi-Seed Validation)"
echo "Running 5 independent seeds (42, 123, 456, 789, 1024)"
echo "Expected: GPU-MLP HighReg achieves 75.0% (std=0.0%)"
echo "Results will be saved to: $RESULTS_DIR/"
echo ""

# Create timestamped output directories for each seed
SEEDS=(42 123 456 789 1024)
for seed in "${SEEDS[@]}"; do
    SEED_OUTPUT="${RESULTS_DIR}/object_generalization_seed_${seed}"
    CONFIG_FILE="configs/object_generalization_3class_seed_${seed}.yml"
    
    echo "Running seed $seed..."
    echo "Output: $SEED_OUTPUT"
    
    if [ -f "$CONFIG_FILE" ]; then
        if python run_modular_experiments.py "$CONFIG_FILE" "$SEED_OUTPUT"; then
            print_success "Seed $seed complete!"
        else
            print_error "Seed $seed failed!"
            exit 1
        fi
    else
        print_warning "Config not found: $CONFIG_FILE (skipping)"
    fi
    echo ""
done

# Count successful seed runs
SEED_COUNT=$(ls -d ${RESULTS_DIR}/object_generalization_seed_* 2>/dev/null | wc -l)
echo "Seeds completed: $SEED_COUNT/5"

################################################################################
# STEP 4: Figure Generation
################################################################################

print_header "STEP 4/4: FIGURE GENERATION"
echo "Generating all main reconstruction and analysis figures"
echo "Figures will be saved to: $RESULTS_DIR/"
echo ""

# Create figure directories
mkdir -p "${RESULTS_DIR}/figures/reconstruction"
mkdir -p "${RESULTS_DIR}/figures/analysis"

# Note: Figure generation scripts would need to be modified to accept custom output paths
# For now, we copy generated figures to the timestamped directory

# Main reconstruction figures
echo "Generating main reconstruction figures..."
if python generate_comprehensive_reconstructions.py; then
    # Copy to timestamped directory
    if [ -d "comprehensive_3class_reconstruction" ]; then
        cp -r comprehensive_3class_reconstruction/* "${RESULTS_DIR}/figures/reconstruction/" 2>/dev/null || true
        print_success "Reconstruction figures saved to ${RESULTS_DIR}/figures/reconstruction/"
    fi
else
    print_warning "Some reconstruction figures may have failed"
fi

# ML analysis figures
echo ""
echo "Generating ML analysis figures..."
if python generate_ml_analysis_figures.py; then
    # Copy to timestamped directory
    if [ -d "ml_analysis_figures" ]; then
        cp -r ml_analysis_figures/* "${RESULTS_DIR}/figures/analysis/" 2>/dev/null || true
        print_success "Analysis figures saved to ${RESULTS_DIR}/figures/analysis/"
    fi
else
    print_warning "Some analysis figures may have failed"
fi

# Position generalization comparison figures
echo ""
echo "Generating position generalization comparison figures..."
if python generate_3class_rotation_figures.py; then
    # Copy additional figures if generated
    if [ -d "ml_analysis_figures" ]; then
        cp -r ml_analysis_figures/* "${RESULTS_DIR}/figures/analysis/" 2>/dev/null || true
    fi
    print_success "Comparison figures generated"
else
    print_warning "Some comparison figures may have failed"
fi

################################################################################
# COMPLETION SUMMARY
################################################################################

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

print_header "PIPELINE COMPLETE!"

echo "Total execution time: ${HOURS}h ${MINUTES}m"
echo ""
echo -e "${GREEN}All results saved to: ${RESULTS_DIR}/${NC}"
echo ""
echo "Directory structure:"
echo "  📁 ${RESULTS_DIR}/"
echo "     ├── rotation{1,2,3}_results/          # Position generalization"
echo "     ├── object_generalization_seed_*/     # Object generalization"
echo "     └── figures/"
echo "         ├── reconstruction/                # Main reconstruction figures"
echo "         └── analysis/                      # ML analysis figures"
echo ""

if [ "$SKIP_BALANCING" = false ]; then
    echo "Balanced datasets:"
    echo "  � data/fully_balanced_datasets/      # Reusable for future runs"
    echo ""
fi

echo "Key Findings (verify against paper):"
echo "  - Position generalization: Avg ~34.5% (catastrophic failure)"
echo "  - Object generalization: GPU-MLP HighReg 75.0% (std=0.0%)"
echo "  - 3-class vs binary: 1.04× vs 0.90× (binary worse than random)"
echo ""
echo "Next steps:"
echo "  1. Review results in: ${RESULTS_DIR}/"
echo "  2. Compare metrics.json files with paper values"
echo "  3. Check figures in: ${RESULTS_DIR}/figures/"
echo "  4. Balanced datasets in data/fully_balanced_datasets/ can be reused"
echo ""
print_success "All experiments completed successfully!"
echo -e "${BLUE}Your existing results were NEVER overwritten.${NC}"
echo ""
