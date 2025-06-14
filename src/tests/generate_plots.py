from modules import *
from functions.exploratory import *
folders = [f for f in os.listdir('../out/sims') if os.path.isdir(os.path.join('../out/sims', f))]
for folder in folders:
    print(folder); plot(folder)
    plt.savefig(f'..figures/{folder}.png', dpi=300, bbox_inches='tight')