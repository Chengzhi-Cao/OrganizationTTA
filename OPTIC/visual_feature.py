





######################################################
_fea = skip1[0,:,:,:].cpu().data.numpy()
_fea_mean = np.mean(_fea, axis=0)
_a = np.clip(_fea_mean, 0, 1) # 将numpy数组约束在[0, 1]范围内
trans_prob_mat = (_a.T/np.sum(_a, 1)).T
df = pd.DataFrame(trans_prob_mat)
plt.figure()
ax = sns.heatmap(df, cmap='jet', cbar=False)
plt.xticks(alpha=0)
plt.tick_params(axis='x', width=0)
plt.yticks(alpha=0)
plt.tick_params(axis='y', width=0)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)

img_path = '{}_img_fea_avenage.jpg'.format(self.global_num)
plt.savefig(os.path.join(self.save_path1,img_path), transparent=True)   
    
for k in range(len(_fea)):
    fea_1 = _fea[k,:,:]

    _a = np.clip(fea_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
    trans_prob_mat = (_a.T/np.sum(_a, 1)).T
    df = pd.DataFrame(trans_prob_mat)
    plt.figure()
    ax = sns.heatmap(df, cmap='jet', cbar=False)
    plt.xticks(alpha=0)
    plt.tick_params(axis='x', width=0)
    plt.yticks(alpha=0)
    plt.tick_params(axis='y', width=0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    img_path = '{}_img_fea_{}.jpg'.format(self.global_num,k)
    plt.savefig(os.path.join(self.save_path1,img_path), transparent=True)   
