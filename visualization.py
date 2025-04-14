




skin_df = pd.read_csv('HAM10000_metadata.csv')


# Data distribution visualization
def distribution_data():
  fig = plt.figure(figsize=(20,8))
  
  ax1 = fig.add_subplot(221)
  skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
  ax1.set_ylabel('Count')
  ax1.set_title('Cell Type')
  
  ax2 = fig.add_subplot(222)
  skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
  ax2.set_ylabel('Count', size=15)
  ax2.set_title('Sex')
  
  ax3 = fig.add_subplot(223)
  skin_df['localization'].value_counts().plot(kind='bar')
  ax3.set_ylabel('Count',size=12)
  ax3.set_title('Localization')
  
  ax4 = fig.add_subplot(224)
  sample_age = skin_df[pd.notnull(skin_df['age'])]
  sns.distplot(sample_age['age'], fit=stats.norm, color='red')
  # sns.histplot(sample_age['age'], color='blue')
  ax4.set_title('Age')
  
  plt.tight_layout()
  plt.show()


def cat_labels():
  n_samples = 5 
  
  fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
  for n_axs, (type_name, type_rows) in zip(m_axs, 
                                           skin_df_balanced.sort_values(['dx']).groupby('dx')):
      n_axs[0].set_title(type_name)
      for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
          c_ax.imshow(c_row['image'])
          c_ax.axis('off')
