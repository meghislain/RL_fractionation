import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

def calculate_effective_dose(volumes, doses, n):
    
    # Calculating the gEUD
    gEUD = sum(v * (d**(1/n)) for v, d in zip(volumes, doses))
    gEUD = gEUD**n
    
    return gEUD

def Equiv_Dose(D, d, ratio):
    """

    Parameters
    ----------
    D : float
        Total treatment dose.
    d : float
        Dose per fraction.
    ratio : float
        alpha/beta ratio.

    Returns
    -------
    EQD2.

    """
    
    return D * ((d+ratio)/(2+ratio))

def Lyman_Kutcher_Burman(gEUD, TD50, m, dx=0.01):
    
    #Lyman-Kutcher-Burman Model
    # uses gEUD in LKB model
    
    t=(gEUD-TD50)/(m*TD50)
    print(f"Integral from -999 to {t}")
    num_range=np.arange(-999,t,dx)
    sum_ntcp=0.
    for idx in range(len(num_range)):
        sum_ntcp+=np.exp(-1*num_range[idx]**2/2)*dx
        
    return 1./np.sqrt(2*np.pi)*sum_ntcp

def axes_off(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.tick_params(axis='both', which='both', length=0)
  
  return ax


"""
df = pd.read_csv('parotid_xerostomia.csv')

fig, ax = plt.subplots(1, 1, figsize = (16,12))
ax = axes_off(ax)
plt.scatter(df['m'], df['TD50'], label="Studies")

for i in range(df.shape[0]):
    if df['authors'].iloc[i] == 'Houweling et al':
        plt.text(df['m'].iloc[i]-0.015, df['TD50'].iloc[i] - 1., df['authors'].iloc[i] + f"\n {df['year'].iloc[i]}", fontsize=18)
    elif (df['authors'].iloc[i] == 'Dijkema et al') and (df['year'].iloc[i] == 2010):
        plt.text(df['m'].iloc[i]-0.015, df['TD50'].iloc[i] - 1., df['authors'].iloc[i] + f"\n {df['year'].iloc[i]}", fontsize=18)
    else:
        plt.text(df['m'].iloc[i] + 0.005, df['TD50'].iloc[i] - 0.5, df['authors'].iloc[i] + f"\n {df['year'].iloc[i]}", fontsize=18)

tmp = df[1:].mean()
plt.scatter(tmp['m'], tmp['TD50'], s=200, label="Chosen value")
#plt.text(tmp['m'] + 0.005, tmp['TD50'] + 0.05, f'Chosen value', fontsize=20)

plt.xlabel('\n m value', fontsize=28)
plt.ylabel('$TD_{50}$ (Gy) \n', fontsize=28)
plt.title('Parotid : Xerostomia \n <=25% saliva after 12 months \n', fontsize=32)
plt.grid(axis='y', alpha=0.5)
plt.legend(fontsize=28, loc=4)

x_ticks = ax.get_xticks().tolist()
if tmp['m'] not in x_ticks:
    x_ticks.append(tmp['m'])
    x_ticks.remove(0.4)
    x_ticks = x_ticks[1:]
    ax.set_xticks(x_ticks)

y_ticks = ax.get_yticks().tolist()
if tmp['TD50'] not in y_ticks:
    y_ticks.append(tmp['TD50'])
    y_ticks.remove(40.0)
    y_ticks = y_ticks[1:]
    ax.set_yticks(y_ticks)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.savefig("figures/parotid_xerostomia.svg", bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (16,12))
ax = axes_off(ax)
gEUD = np.arange(0,100,0.1)

for i in range(df.shape[0]):
    lst = []
    for j in range(len(gEUD)):
        lst.append(Lyman_Kutcher_Burman(gEUD[j], df['TD50'].iloc[i], df['m'].iloc[i]))
    plt.plot(gEUD, np.array(lst)*100, label = str(df['year'].iloc[i]) + ', ' + df['authors'].iloc[i])


plt.xlabel('\n gEUD (Gy)', fontsize=28)
plt.ylabel('NTCP (%) \n', fontsize=28)
plt.title('Parotid : Xerostomia \n <=25% saliva after 12 months \n', fontsize=32)
plt.grid(axis='y', alpha=0.5)
plt.legend(fontsize=28, loc=4)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig('figures/NTCP_parotid.svg')
plt.show()
"""


df = pd.read_csv('lung_pneumonitis.csv')


fig, ax = plt.subplots(1, 1, figsize = (16,12))
ax = axes_off(ax)
plt.scatter(df['m'], df['TD50'])

for i in range(df.shape[0]):
    plt.text(df['m'].iloc[i] + 0.005, df['TD50'].iloc[i] + 0.01, df['authors'].iloc[i] + f"\n {df['year'].iloc[i]}", fontsize=18)

plt.xlabel('\n m value', fontsize=28)
plt.ylabel('TD50 \n', fontsize=28)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.title('Lung : Pneumonitis \n', fontsize=32)
plt.grid(axis='y', alpha=0.5)
plt.savefig("figures/lung_pneumonitis.svg", bbox_inches='tight')
plt.show()




"""
df = pd.read_csv('rectum_rectal_bleeding.csv')


fig, ax = plt.subplots(1, 1, figsize = (16,12))
ax = axes_off(ax)
plt.scatter(df['m'], df['TD50'])

for i in range(df.shape[0]):
    plt.text(df['m'].iloc[i] + 0.001, df['TD50'].iloc[i] + 0.01, df['authors'].iloc[i] + f"\n {df['year'].iloc[i]}", fontsize=18)

plt.scatter(0.15, 80.1, s=200, label="Chosen value")
plt.text(0.15 + 0.0025, 80.1 + 0.05, f'Chosen value', fontsize=20)

plt.xlabel('\n m value', fontsize=28)
plt.ylabel('TD50 \n', fontsize=28)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.title('Rectum : rectal bleeding \n', fontsize=32)
plt.grid(axis='y', alpha=0.5)
plt.savefig("figures/rectum_rectal_bleeding.svg", bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (16,12))
ax = axes_off(ax)
gEUD = np.arange(0,120,1)

for i in range(df.shape[0]):
    lst = []
    for j in range(len(gEUD)):
        lst.append(Lyman_Kutcher_Burman(gEUD[j], df['TD50'].iloc[i], df['m'].iloc[i]))
    plt.plot(gEUD, np.array(lst)*100, label = str(df['year'].iloc[i]) + ', ' + df['authors'].iloc[i])


plt.xlabel('\n gEUD (Gy)', fontsize=28)
plt.ylabel('NTCP (%) \n', fontsize=28)
plt.title('Rectum : rectal bleeding \n', fontsize=32)
plt.grid(axis='y', alpha=0.5)
plt.legend(fontsize=28, loc='best')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig('figures/NTCP_rectum.svg')
plt.show()
"""
"""
df_ = pd.read_csv('alpha_beta_parameters.csv')
df = df_[df_["organ"] == "parotid"]

fig, ax = plt.subplots(1, 1, figsize = (16,12))
ax = axes_off(ax)
plt.scatter(df['alpha'], df['beta'])

plt.xlabel('\n alpha', fontsize=28)
plt.ylabel('beta \n', fontsize=28)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.title('Parotid \n', fontsize=32)
plt.grid(axis='y', alpha=0.5)
plt.savefig("figures/alpha_beta_parameters_parotid.svg", bbox_inches='tight')
plt.show()
"""