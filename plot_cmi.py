import numpy as np
from scipy import sparse
from functools import partial
import math
import shelve
from matplotlib import pyplot as plt
import scipy


import os, time
import numpy as np
from threadpoolctl import threadpool_limits



# with shelve.open("data/zxz_gapless_model/set_6/result") as db:
#     h_list = db["h_list"]
#     order_parameter_list = db["order_parameter_list"]
#     corr_list = db["corr_list"]
#
# with shelve.open("data/zxz_gapless_model/set_5/result") as db:
#     corr_list_1 = db["corr_list"]
#
# print(h_list)
# print(corr_list)
# plt.plot(h_list, np.abs(corr_list), "o-", linewidth=0.8, markersize=2)
# plt.plot(h_list, np.abs(order_parameter_list), "o-", linewidth=0.8, markersize=2)
# plt.plot(h_list, np.abs(corr_list_1), "o-", linewidth=0.8, markersize=2)
# #plt.plot(h_list, order_parameter_list, "o-", linewidth=0.8, markersize=2)
#
# plt.xlabel("h")
# plt.savefig("plot.pdf")



# with shelve.open("data/zxz_gapless_model/set_11/result") as db:
#     corr_list_0 = db["corr_list"]
# with shelve.open("data/zxz_gapless_model/set_10/result") as db:
#     corr_list_1 = db["corr_list"]
# with shelve.open("data/zxz_gapless_model/set_9/result") as db:
#     corr_list_2 = db["corr_list"]
# with shelve.open("data/zxz_gapless_model/set_12/result") as db:
#     corr_list_3 = db["corr_list"]
# with shelve.open("data/zxz_gapless_model/set_13/result") as db:
#     corr_list_4 = db["corr_list"]
#
#
# p_list = np.arange(21) * 0.025
# plt.plot(p_list, np.abs(corr_list_0), "o-", linewidth=0.8, markersize=2)
# plt.plot(p_list, np.abs(corr_list_1), "o-", linewidth=0.8, markersize=2)
# plt.plot(p_list, np.abs(corr_list_2), "o-", linewidth=0.8, markersize=2)
# plt.plot(p_list, np.abs(corr_list_3), "o-", linewidth=0.8, markersize=2)
# plt.plot(p_list, np.abs(corr_list_4), "o-", linewidth=0.8, markersize=2)
# plt.xlabel("p")
# plt.savefig("plot.pdf")



# import shutil
#
# src = "data/zxz_gapless_model/tenpy_result_0"
# N = 21  # change to however many copies you want
# for i in range(1, N):
#     dst = f"data/zxz_gapless_model/tenpy_result_{i}"
#     shutil.copy(src, dst)




# x = np.log(1-np.array(h_list[:-7]))
# y = np.log(corr_list[:-7])
# plt.plot(x, y, "o", linewidth=0.8, markersize=2)
#
# a, b = np.polyfit(x, y, 1)
# print("slope a =", a)
# print("intercept b =", b)
# plt.plot([x[0], x[-1]], [a * x[0] + b, a * x[-1] + b], "-", color="C0", linewidth=0.8, markersize=2)
# plt.savefig("plot.pdf")

#with shelve.open("data/zxz_model/result") as db:
#    for key in db:
#        print(key, db[key])

# with shelve.open("data/zxz_model/result") as db:
#     h_list = db["h_list"]
# with shelve.open("data/zxz_model/result_strange_correlator") as db:
#     strange_corr_list = db["strange_corr_list"]
#     type_2_corr_list = db["type_2_corr_list"]
# fig, ax1 = plt.subplots()
#
# # First y-axis
# ax1.plot(h_list, strange_corr_list, '-', label='Strange Correlator', color='tab:blue')
# ax1.set_xlabel('h')
# ax1.set_ylabel('Strange Correlator', color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')
#
# # Second y-axis sharing the same x-axis
# ax2 = ax1.twinx()
# ax2.plot(h_list, type_2_corr_list, '-', label='Type-2 Strange Correlator', color='tab:orange')
# ax2.set_ylabel('Type-2 Correlator', color='tab:orange')
# ax2.tick_params(axis='y', labelcolor='tab:orange')
#
# # Optional: improve layout and legend
# fig.tight_layout()
# plt.savefig("plot.pdf")


cmi_list = []
negativity_list = []
for i in [0, 1, 2, 3]:
    with shelve.open("data/zxz_model/set_" + str(i) + "/cmi_data") as db:
        cmi_list += db["cmi_list"]
        negativity_list += db["negativity_list"]
h_list = [0 + 0.1 *i for i in range(16)] + [0.9 + 0.01 * i for i in range(16)]
print(cmi_list)
print(negativity_list)

h_list, cmi_list, negativity_list = zip(
    *sorted(zip(h_list, cmi_list, negativity_list))
)

plt.rcParams.update({
    "font.size": 22,           # default text
    "axes.labelsize": 20,      # x/y label font
    "xtick.labelsize": 16,     # tick label font
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

plt.plot(h_list, cmi_list, "o-", linewidth=0.8, markersize=2)
plt.xlabel('$h$') # , fontsize=22
plt.ylabel('$I(A:B|C)$')
plt.tight_layout()
plt.savefig("zxz_cmi.pdf")

plt.clf()

plt.plot(h_list, negativity_list, "o-", linewidth=0.8, markersize=2)
plt.xlabel('$h$') # , fontsize=22
plt.ylabel(r'$\mathcal{N}(\rho)$')
plt.tight_layout()
plt.savefig("zxz_negativity.pdf")