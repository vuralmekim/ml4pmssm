# ML4pMSSM

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ML4PMSSM.git
   ```
2. Run
   ```bash
   cd ml4pmssm
   conda env create -f environment.yml
   conda activate ml4pmssm
   chmod +x train_and_plot.sh
   ./train_and_plot.sh <input root file> <config.yaml>
   ```
using the config yaml it trains, saves the model and train-val losses, evaluates, saves the predictions, produces loss function and a 2d histogram for predictions - actual values
