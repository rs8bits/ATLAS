from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Disk Type Data Structure
# =========================
@dataclass
class DiskType:
    name: str      # Disk name
    W: float       # Write throughput as log disk (ops/s)
    R: float       # Read throughput as data disk (ops/s)
    M: float       # Price per GB

# =========================
# Cost Model
# =========================
def cost_of_configuration(C_server, V_i, V_j, M_i, M_j):
    """
    Compute total cost of disk configuration D_i D_j.
    C_total = server cost + capacity_cost_log + capacity_cost_data
    """
    return C_server + V_i * M_i + V_j * M_j

# =========================
# Performance Model
# =========================
def throughput_of_configuration(W_i, R_j, r):
    """
    Compute throughput under given read ratio r.
    P(r) = 1 / (r / R_j + (1 - r) / W_i)
    """
    return 1.0 / (r / R_j + (1 - r) / W_i)

# =========================
# Cost-Performance Ratio
# =========================
def cpr(P, C_total):
    """
    CPR = throughput / total cost
    """
    return P / C_total

# =========================
# Evaluate All Configurations
# =========================
def evaluate_all_configs(disks, C_server, V_i, V_j, r_values):
    """
    Evaluate all disk pairs (D_i, D_j) over all read ratios r.
    Return a Pandas DataFrame with throughput, cost, and CPR.
    """
    results = []
    for (i, disk_i), (j, disk_j) in itertools.product(enumerate(disks), repeat=2):
        for r in r_values:
            P = throughput_of_configuration(disk_i.W, disk_j.R, r)
            C_total = cost_of_configuration(C_server, V_i, V_j, disk_i.M, disk_j.M)
            CPR = cpr(P, C_total)
            results.append({
                "log_disk": disk_i.name,
                "data_disk": disk_j.name,
                "r": r,
                "P_ops_per_s": P,
                "C_total": C_total,
                "CPR": CPR
            })
    return pd.DataFrame(results)

# =========================
# Universal CPR (unknown workload)
# =========================
def compute_ucpr(disks, C_server, V_i, V_j):
    """
    Compute the Universal CPR (UCPR) under unknown workload.
    UCPR = ∫ CPR(r) dr over r∈[0,1], closed-form
    """
    results = []
    for disk_i, disk_j in itertools.product(disks, repeat=2):
        C_total = cost_of_configuration(C_server, V_i, V_j, disk_i.M, disk_j.M)
        W_i, R_j = disk_i.W, disk_j.R
        if W_i == R_j:
            ucpr_value = W_i / C_total
        else:
            ucpr_value = (W_i * R_j) / (W_i - R_j) * np.log(W_i / R_j) / C_total
        results.append({
            "log_disk": disk_i.name,
            "data_disk": disk_j.name,
            "UCPR": ucpr_value
        })
    return pd.DataFrame(results)

# =========================
# Main Program
# =========================
if __name__ == "__main__":
    # ====== Example Disk Configurations ======
    disks = [
        DiskType("HDD", 30000, 81000, 0.00049),
        # DiskType("HDD-high", 32000, 81500, 0.0014),
        # DiskType("SSD-general", 34000, 82000, 0.00097),
        # DiskType("SSD-general-V2", 35000, 83000, 0.00098),
        DiskType("SSD", 39900, 89000, 0.0042)
    ]

    # ====== System Parameters ======
    C_server = 3.696     # Base server cost (USD)
    V_i = 150            # Log disk capacity (GB)
    V_j = 150            # Data disk capacity (GB)
    r_values = [0.0, 0.5, 0.95,1.0]  # Read ratio

    # ====== Font Settings for English Output ======
    plt.rcParams['axes.unicode_minus'] = False

    # ====== Evaluate CPR ======
    df = evaluate_all_configs(disks, C_server, V_i, V_j, r_values)
    df.to_csv("evaluation_all_configs.csv", index=False, encoding="utf-8-sig")
    print("Results saved to evaluation_all_configs.csv")

    # ====== Compute UCPR ======
    ucpr_df = compute_ucpr(disks, C_server, V_i, V_j)
    print("\nUniversal CPR (UCPR) under unknown workload:")
    print(ucpr_df.sort_values("UCPR", ascending=False))

    # ====== Visualize Optimal CPR ======
    best_configs = df.loc[df.groupby("r")["CPR"].idxmax()]
    plt.figure(figsize=(8, 5))
    plt.plot(best_configs["r"], best_configs["CPR"], marker="o")
    plt.title("Optimal CPR Across Read Ratios", fontsize=14)
    plt.xlabel("Read Ratio (r)")
    plt.ylabel("CPR (throughput / cost)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ====== Visualize Optimal Disk Combination ======
    plt.figure(figsize=(8, 5))
    plt.plot(best_configs["r"], best_configs["log_disk"] + "-" + best_configs["data_disk"], marker="s")
    plt.title("Optimal Disk Combination Across Read Ratios", fontsize=14)
    plt.xlabel("Read Ratio (r)")
    plt.ylabel("Best Disk Pair (log-data)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


