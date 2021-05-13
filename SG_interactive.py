import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import scipy as sp
from scipy import constants, stats
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, TextBox, RadioButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable

""" NOTES/ACKNOWLEDGMENTS
Inspiration and certain formulae/parameters (e.g. the magnetic field gaussian shape) from Dr. Ahmet Bingul's 2008 code.

This certainly works on Mac OS 10.14, however other platforms are untested.
"""

#init window setup
plt.rcParams['toolbar'] = 'None'
fig, ax = plt.subplots(figsize=(0.1,0.1), sharex=True)
plt.subplots_adjust(bottom=0.4)
manager = plt.get_current_fig_manager()
manager.window.setFixedSize(700,700)
manager.set_window_title('SG Simulator')

#original experiment/default parameters:
L = 0.035 #length of magneton, m
D = 0.035 #magnet-to-plate distance, m
x_max = 8*10**(-4) #aperture x-size, m
z_max = 3.5*10**(-5) #aperture z-size, m
B = 0.1 #base field strength, T
T = 1300 #oven temperature, K
l = 0.5 #initial spin

#constants/fundamental parameters
hbar = constants.hbar
uB = constants.value('Bohr magneton')
g = constants.value('electron g factor')
u = constants.value('atomic mass constant')
m_silver = 107.87 * u #silver mass, kg
m_e = constants.m_e #electron mass, kg
k = constants.k #Boltzmann constant, J K^-1
e = constants.e #electron charge, C

#other variable/plot initialization
spins = []
x_pos = []
z_final = []

quantum_bool = 1
spin_mag = np.sqrt(l * (l+1)) * hbar
n_atoms = 10000

#scatter plot init
patternPlot = plt.scatter(x_pos, z_final, s=1)

#histograms
divider = make_axes_locatable(ax)
ax_histx = divider.append_axes("top", 0.8, pad=0.1, sharex=ax)
ax_histz = divider.append_axes("right", 0.8, pad=0.1, sharey=ax)
ax_histx.xaxis.set_tick_params(labelbottom=False)
ax_histz.yaxis.set_tick_params(labelleft=False)

#sliders
axL = plt.axes([0.5, 0.28, 0.38, 0.03])
axD = plt.axes([0.5, 0.24, 0.38, 0.03])
axB = plt.axes([0.5, 0.20, 0.38, 0.03])

sL = Slider(axL, r"$L$", 0.001, 1, valinit = L)
sD = Slider(axD, r"$D$", 0.001, 1, valinit = D)
sB = Slider(axB, r"$B_0$", 0.01, 10, valinit = B)

sL.valtext.set_text(('{:.0f} cm').format(L*100))
sD.valtext.set_text(('{:.0f} cm').format(D*100))
sB.valtext.set_text(('{:.1f} T').format(B))

#main plotting function
def update(val):
    #clear all
    x_pos = []
    z_final = []
    velocities = []
    spins = []

    #update user-input parameters
    n_atoms = int(natoms_box.text)
    L = sL.val
    D = sD.val
    B = sB.val

    #update sliders
    sL.valtext.set_text(('{:.0f} cm').format(L*100))
    sD.valtext.set_text(('{:.0f} cm').format(D*100))
    sB.valtext.set_text(('{:.1f} T').format(B))

    #set quantum_bool
    if quantRadio.value_selected == "Classical":
        quantum_bool = 0
    else:
        quantum_bool = 1

    #set spins
    if spinRadio.value_selected == "Spin-1":
        l = 1
    elif spinRadio.value_selected == r"Spin-$\frac{1}{2}$":
        l = 0.5
    else:
        l = 1.5

    spin_mag = np.sqrt(l * (l+1)) * hbar

    #generate distributions
    maxwell = stats.maxwell
    uniform = stats.uniform

    #Maxwell-Boltzmann velocity distribution
    velocities = maxwell.rvs(loc=0, scale=np.sqrt(k*T/m_silver), size=n_atoms)

    #assign spins
    if quantum_bool == 0: # classical
        for i in uniform.rvs(size=n_atoms):
            theta = 2 * np.pi * i
            spins.append(spin_mag * np.cos(theta))
    else: # quantum
        for i in uniform.rvs(size=n_atoms):
            if l == 0.5:
                if i < 0.5:
                    spins.append(hbar / 2)
                else:
                    spins.append(-hbar / 2)
            elif l == 1:
                if i < 1/3:
                    spins.append(hbar)
                elif i >= 1/3 and i < 2/3:
                    spins.append(0)
                else:
                    spins.append(-hbar)
            elif l == 1.5:
                if i < 0.25:
                    spins.append(3 * hbar / 2)
                elif i >= 0.25 and i < 0.5:
                    spins.append(hbar / 2)
                elif i >= 0.5 and i < 0.75:
                    spins.append(-hbar / 2)
                else:
                    spins.append(-3 * hbar / 2)

    #random initial positions in oven aperture
    x_pos = uniform.rvs(loc=-x_max/2, scale=x_max, size=n_atoms)
    z_pos = uniform.rvs(loc=-z_max/2, scale=z_max, size=n_atoms)

    #calculate field effects
    for i in range(len(spins)):
        if magRadio.value_selected == "Field On":
            if fieldRadio.value_selected == fieldLabels[0]:
                dB = 0
            elif fieldRadio.value_selected == fieldLabels[1]:
                dB = B
            elif fieldRadio.value_selected == fieldLabels[2]:
                dB = B * np.exp(-25 * x_pos[i]**2 / x_max**2)
        else:
            dB = 0

        #acceleration and final position calculation
        uz = - spins[i] * g * e / (2 * m_e)
        acceleration = uz * dB / m_silver
        z = z_pos[i] + (1/2) * acceleration * (L / velocities[i])**2 + np.sign(acceleration) * D * np.sqrt(2 * np.abs(acceleration) * L) / velocities[i]
        z_final.append(z)

    #histogram updates
    ax_histx.cla()
    ax_histz.cla()

    x_pos_mm = np.multiply(x_pos, 1000)
    z_final_mm = np.multiply(z_final, 1000)

    ax_xlim = 3 * np.std(x_pos_mm)
    ax_ylim = 3 * np.std(z_final_mm)

    xbinwidth = (np.max(x_pos_mm) - np.min(x_pos_mm)) / 10
    zbinwidth = (np.max(z_final_mm) - np.min(z_final_mm)) / 10
    xbins = np.arange(-ax_xlim, ax_xlim, ax_xlim/10)
    zbins = np.arange(-ax_ylim, ax_ylim, ax_ylim/10)

    ax_histx.hist(x_pos_mm, bins=xbins)
    ax_histz.hist(z_final_mm, bins=zbins, orientation='horizontal')

    #set axis limits
    ax.set_xlim(-ax_xlim, ax_xlim)
    ax.set_ylim(-ax_ylim, ax_ylim)

    #plot points
    patternPlot.set_offsets(np.c_[x_pos_mm, z_final_mm])

    fig.canvas.draw_idle()

#quantum toggle
quantRadioax = plt.axes([0.175, 0.02, 0.14, 0.14], aspect='equal')
quantRadio = RadioButtons(quantRadioax, ("Quantum", "Classical"), activecolor="C0")
for circle in quantRadio.circles:
    circle.set_radius(0.06)

quantRadio.on_clicked(update)

#magnetic field toggle
magRadioax = plt.axes([0.32, 0.02, 0.14, 0.14], aspect='equal')
magRadio = RadioButtons(magRadioax, ("Field On", "Field Off"), activecolor="C0")
for circle in magRadio.circles:
    circle.set_radius(0.06)

magRadio.on_clicked(update)

#field profile options
fieldRadioDrawax = plt.axes([0.465, 0.02, 0.42, 0.14])
fieldRadioDraw = RadioButtons(fieldRadioDrawax, '')
fieldRadioax = plt.axes([0.465, 0.02, 0.14, 0.14], frameon=False, aspect='equal')
fieldLabels = [r"Uniform ($\frac{\partial B_z}{\partial z}=0$)", r"Constant Gradient ($\frac{\partial B_z}{\partial z}=B_0$)", r"x-Dependent Gaussian ($\frac{\partial B_z}{\partial z}=B_0e^{-25x^2}$)"]
fieldRadio = RadioButtons(fieldRadioax, (fieldLabels[0], fieldLabels[1], fieldLabels[2]), activecolor="C0")
for circle in fieldRadio.circles:
    circle.set_radius(0.06)

fieldRadio.on_clicked(update)

#spin options
spinRadioax = plt.axes([0.175, 0.17, 0.14, 0.14], aspect='equal')
spinRadio = RadioButtons(spinRadioax, (r"Spin-$\frac{1}{2}$", "Spin-1", r"Spin-$\frac{3}{2}$"), activecolor="C0")
for circle in spinRadio.circles:
    circle.set_radius(0.06)

spinRadio.on_clicked(update)

#n_atoms input
natomsDrawax = plt.axes([0.32, 0.17, 0.14, 0.14])
fieldRadioDraw = RadioButtons(natomsDrawax, '')
natomsDrawax.text(0.5, 0.6, "Number of\nAtoms", ha="center")
natomsax = plt.axes([0.34, 0.18, 0.1, 0.05])
natoms_box = TextBox(natomsax, "", initial="10000")
natoms_box.on_submit(update)

#call update on slider changes
sL.on_changed(update)
sD.on_changed(update)
sB.on_changed(update)

#axis labels
ax.set_xlabel("x (mm)")
ax.set_ylabel("z (mm)")

#display
plt.show()
