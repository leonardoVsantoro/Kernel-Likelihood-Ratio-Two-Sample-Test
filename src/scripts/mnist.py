# from modules import *
# from functions import *
# ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
# os.makedirs(f'../out/mnist/{ts}', exist_ok=True)


# NUM_CORES = 72
# num_replications = 200
# num_samples = 75
# num_permutations = 250
# test_names = ['*KL*','*HS*','*CM*', 'MMD', 'KNN', 'FR', 'HT']
# reg_factor = 1e-4
# kernel = 'sqeuclidean'

# # -------------------------------- import data ---------------------------------------------------------------------------------
# mnist_data_train = pd.read_csv('../datasets/MNIST/mnist_train.csv').set_index('label')
# mnist_data_train /= 255

# # -------------------------------- plot sample data: original and perturbed -----------------------------------------------------
# for i in range(3):
#     fig, axs = plt.subplots(figsize=(8, 3), nrows=2, ncols=5)
#     for num, ax in enumerate(axs.ravel()):
#         X = mnist_data_train.loc[num].values[0]

#         im = X.reshape(28, 28)
#         im_white = (X + np.random.normal(0, .4, X.shape)).reshape(28, 28)
#         im_blur = gaussian_filter(X.reshape(28, 28), 2)

#         sns.heatmap([im, im_white, im_blur][i], square=True, cbar=False, ax=ax, vmin=-.2, vmax=1.2)
#         ax.axis('off')
#     fig.savefig(f'../out/mnist/{["original", "white_noise", "blur"][i]}.png')

# # -------------------------------- comparing 2 sets of digits -----------------------------------------------------------------
# group_1 = [9, 6, 8]
# group_2 = [4, 8]
# XYpairs = [(mnist_data_train.loc[group_1].sample(num_samples).values,
#             mnist_data_train.loc[group_2].sample(num_samples).values,
#             ) for _ in range(num_replications)]

# # --------------------------------additive Gaussian white noise -----------------------------------------------------------------
# sigmas = np.linspace(0.5, 1.5, 5)
# results = {}
# for sigma in sigmas:
#     iter_args = [(X + np.random.normal(0, sigma, X.shape),
#                   Y + np.random.normal(0, sigma, Y.shape),
#                   num_permutations, test_names, kernel, reg_factor)
#                  for (X, Y) in XYpairs]

#     results[sigma] = Parallel(n_jobs=NUM_CORES)(
#         delayed(run_iteration_test)(*args) for args in iter_args
#     )
# data = []
# for sigma in sigmas:
#     for line in results[sigma]:
#         for el in line:
#             test_name, value = el
#             data.append({"sigma": sigma, "test_name": test_name, "value": value})
# df = pd.DataFrame(data)
# rej_perc_df = df.groupby(["sigma", "test_name"])["value"].mean().reset_index()
# rej_perc_df.to_csv(f'../out/mnist/{ts}/rp_additive.csv', index=False)

# # -------------------------------- Blurring - Gaussian convolution --------------------------------------------------------------
# sigmas = np.linspace(2, 6, 5)
# results = {}
# for sigma in sigmas:
#     iter_args = [(np.array([gaussian_filter((_ + np.random.normal(0, .25, _.shape)).reshape(28, 28), sigma).flatten() for _ in X]),
#                   np.array([gaussian_filter((_ + np.random.normal(0, .25, _.shape)).reshape(28, 28), sigma).flatten() for _ in Y]),
#                   num_permutations, test_names, kernel, reg_factor)
#                  for (X, Y) in XYpairs]

#     results[sigma] = Parallel(n_jobs=NUM_CORES)(
#         delayed(run_iteration_test)(*args) for args in iter_args
#     )
# data = []
# for sigma in sigmas:
#     for line in results[sigma]:
#         for el in line:
#             test_name, value = el
#             data.append({"sigma": sigma, "test_name": test_name, "value": value})
# df = pd.DataFrame(data)
# rej_perc_df = df.groupby(["sigma", "test_name"])["value"].mean().reset_index()
# rej_perc_df.to_csv(f'../out/mnist/{ts}/rp_blurred.csv', index=False)

# # -------------------------------- Plot percentage of rejections vs Noise -----------------------------------------------------
# sns.set_style("whitegrid"); sns.set_palette("bright"); sns.set_theme(font="DejaVu Sans")

# fig, axs = plt.subplots(figsize=(12, 6), ncols=2)
# df_additive = pd.read_csv(f'../out/mnist/{ts}/rp_additive.csv')
# df_blurred = pd.read_csv(f'../out/mnist/{ts}/rp_blurred.csv')

# fig.suptitle('MNIST', fontweight="bold")

# for ax, df, title in zip(axs, [df_additive, df_blurred], ['Additive Noise', 'Blurred Image']):
#     sns.lineplot(data=df, x="sigma", y="value", hue="test_name", style="test_name", marker='o', dashes=True, ax=ax, alpha = .95, lw=1.5)
    
#     ax.set_ylim(0, 1.025)
#     ax.set_xlabel("Sigma")
#     ax.set_ylabel("Rejection Percentage")
#     ax.set_title(title)
#     ax.legend(title="Test Name")
#     # ax.set_aspect('equal', adjustable='datalim')
#     ax.get_legend().remove()
# fig.legend(axs[0].get_legend_handles_labels()[0], axs[0].get_legend_handles_labels()[1], 
#             loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(test_names), fontsize=11)
# sns.despine()
# plt.tight_layout()
# fig.savefig(f'../out/mnist/{ts}/mnist_rp_additive_blurred.png', bbox_inches='tight')


# # -------------------------------- End ---------------------------------------------------------------------------------------
