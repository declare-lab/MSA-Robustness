
# import matplotlib.pyplot as plt

# num_list = [1.5,0.6,7.8,6]
# name_list = ['L', 'V', 'A']

# bars = plt.bar([1,2,3], num_list, color='grey', width=0.2, tick_label=name_list)

# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)))


# # plt.bar(range(len(num_list)), num_list)
# plt.show()

fig1,ax = plt.subplots(figsize=(10,6))

ax.bar(df.index+0.0, df.iloc[:,0],width=0.1,label='BKNG')
ax.bar(df.index+0.1, df.iloc[:,1],width=0.1,label='MCD')
ax.bar(df.index+0.2, df.iloc[:,2],width=0.1,label='YUM')
plt.grid(True)
ax.set_yscale('symlog')
plt.legend()
plt.xlabel('date')
plt.ylabel('value')
plt.title('ROE')