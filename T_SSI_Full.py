import openseespy.opensees as ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, pi
from datetime import datetime
import os
import itertools
import random

class SSI_FrameAnalysis:
    """
    OpenSees model for 2D steel frame with SSI effects
    Machine Learning Dataset Generation - THE ULTIMATE VERSION
    
    Includes:
    1. Parametric Study Engine (Memory Efficient)
    2. Fixed Schema Output (10-Story Padding)
    3. Error Logging
    4. Full Statistical Suite (Report + 12 Visualizations)
    """
    
    def __init__(self):
        self.results = []
        self.output_file = None
        self.header_written = False
        self.error_log_file = "analysis_errors.log"
        self.stats_report_file = "statistical_summary_report.txt"
        
        # --- ACADEMIC PLOT STYLING ---
        self.set_academic_style()

    def set_academic_style(self):
        """Sets matplotlib params for publication-quality figures."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,        # High resolution for print
            'savefig.dpi': 300,
            'axes.linewidth': 1.0,
            'grid.color': '#DDDDDD',  # Subtle grid
            'grid.linestyle': '-',
            'grid.alpha': 0.5,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })

    def calculate_spring_stiffness(self, Gs, nu_s, A, a, b):
        """
        Calculate soil spring stiffnesses using Newmark & Rosenblueth formulas
        """
        sqrt_A = sqrt(A)
        # Geometry factors for rectangular footing
        beta_x = 2.0  # Horizontal
        beta_z = 1.0  # Vertical
        
        Kv = (Gs / (1 - nu_s)) * beta_z * sqrt_A  # Vertical
        Kh = 2 * (1 + nu_s) * Gs * beta_x * sqrt_A  # Horizontal
        Kr = ((1 + nu_s) / 4) * Gs * beta_x * (a**2 + b**2) * sqrt_A  # Rotational
        return Kv, Kh, Kr
    
    def log_error(self, params, error_msg):
        """Logs failed analysis parameters to a text file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.error_log_file, "a") as f:
            f.write(f"[{timestamp}] Error: {error_msg} | Params: {str(params)}\n")

    def create_2D_frame_model(self, params):
        """
        Create 2D steel frame model with SSI springs
        """
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        
        # Extract parameters
        Ns, Nbl = params['Ns'], params['Nbl']
        H, L = params['H'], params['L']
        Ec = params['Ec'] * 1e6
        Ic, Ac = params['Ic'], params['Ac']
        Ib, Ab = params['Ib'], params['Ab']
        rho_c = params['rho_c']
        Vs, rho_s, nu_s = params['Vs'], params['rho_s'], params['nu_s']
        a, b = params['footing_dim']
        include_ssi = params['include_ssi']
        
        Gs = rho_s * Vs**2
        A_footing = a * b
        Kv, Kh, Kr = self.calculate_spring_stiffness(Gs, nu_s, A_footing, a, b)
        
        node_id = 1
        nodes = {}
        
        # Create nodes
        for story in range(Ns + 1):
            for bay in range(Nbl + 1):
                x = bay * L
                y = story * H
                
                if story == 0 and include_ssi:
                    # Base node (fixed)
                    ops.node(node_id, x, y)
                    ops.fix(node_id, 1, 1, 1)
                    nodes[(story, bay, 'base')] = node_id
                    node_id += 1
                    # Struct node (free)
                    ops.node(node_id, x, y)
                    nodes[(story, bay, 'struct')] = node_id
                    node_id += 1
                elif story == 0 and not include_ssi:
                    # Fixed base
                    ops.node(node_id, x, y)
                    ops.fix(node_id, 1, 1, 1)
                    nodes[(story, bay)] = node_id
                    node_id += 1
                else:
                    ops.node(node_id, x, y)
                    nodes[(story, bay)] = node_id
                    node_id += 1
        
        # Materials & Transform
        ops.uniaxialMaterial('Elastic', 1, Ec)
        ops.geomTransf('Linear', 1)
        
        # SSI Springs
        if include_ssi:
            ops.uniaxialMaterial('Elastic', 10, Kh)
            ops.uniaxialMaterial('Elastic', 11, Kv)
            ops.uniaxialMaterial('Elastic', 12, Kr)
            
            ele_id = 1
            for bay in range(Nbl + 1):
                n_base = nodes[(0, bay, 'base')]
                n_struct = nodes[(0, bay, 'struct')]
                ops.element('zeroLength', ele_id, n_base, n_struct, '-mat', 10, 11, 12, '-dir', 1, 2, 3)
                ele_id += 1
        
        # Elements
        ele_id = 100 if include_ssi else 1
        # Columns
        for bay in range(Nbl + 1):
            for story in range(Ns):
                if story == 0 and include_ssi:
                    n_i = nodes[(story, bay, 'struct')]
                    n_j = nodes[(story + 1, bay)]
                else:
                    n_i = nodes[(story, bay)] if not include_ssi or story > 0 else nodes[(story, bay, 'struct')]
                    n_j = nodes[(story + 1, bay)]
                ops.element('elasticBeamColumn', ele_id, n_i, n_j, Ac, Ec, Ic, 1)
                ele_id += 1
                
        # Beams
        for story in range(1, Ns + 1):
            for bay in range(Nbl):
                n_i = nodes[(story, bay)]
                n_j = nodes[(story, bay + 1)]
                ops.element('elasticBeamColumn', ele_id, n_i, n_j, Ab, Ec, Ib, 1)
                ele_id += 1
        
        # Mass
        col_mass = rho_c * Ac * H / 9.81
        beam_mass = rho_c * Ab * L / 9.81
        slab_mass = 3.0 + 0.5 * L + 0.2 * H
        
        for story in range(1, Ns + 1):
            for bay in range(Nbl + 1):
                n_i = nodes[(story, bay)]
                mass = 0.5*col_mass + 0.5*beam_mass if (bay==0 or bay==Nbl) else 0.5*col_mass + beam_mass
                mass += slab_mass
                ops.mass(n_i, mass, mass, 1e-10)
        
        return nodes
    
    def perform_eigenvalue_analysis(self, num_modes=5):
        try:
            eigenvalues = ops.eigen(num_modes)
        except:
            try:
                eigenvalues = ops.eigen('-fullGenLapack', num_modes)
            except:
                return [0.0]*num_modes, [0.0]*num_modes
            
        periods, freqs = [], []
        for lam in eigenvalues:
            if lam > 0:
                T = 2 * pi / sqrt(lam)
                periods.append(T)
                freqs.append(1/T)
            else:
                periods.append(0.0)
                freqs.append(0.0)
        return periods, freqs
    
    def get_mode_shapes(self, nodes, Ns, max_stories, num_modes=3):
        shapes = {}
        try:
            for mode in range(1, num_modes + 1):
                roof_node = nodes.get((Ns, 0))
                scale = 1.0
                if roof_node:
                    roof_disp = ops.nodeEigenvector(roof_node, mode, 1)
                    scale = 1.0 / roof_disp if abs(roof_disp) > 1e-10 else 1.0
                
                for story in range(max_stories + 1):
                    key = f'Mode{mode}_Story{story}'
                    if story <= Ns:
                        if story == 0:
                            node = nodes.get((0, 0, 'struct'), nodes.get((0, 0)))
                        else:
                            node = nodes.get((story, 0))
                        
                        if node:
                            shapes[key] = ops.nodeEigenvector(node, mode, 1) * scale
                        else:
                            shapes[key] = 0.0
                    else:
                        shapes[key] = 0.0
            return shapes
        except:
            return {}
    
    def get_steel_sections(self):
        return {
            'W200x52': {'Ic': 4.57e-5, 'Ac': 0.00664, 'Ib': 4.57e-5, 'Ab': 0.00664},
            'W250x73': {'Ic': 1.05e-4, 'Ac': 0.00932, 'Ib': 1.05e-4, 'Ab': 0.00932},
            'W310x97': {'Ic': 2.22e-4, 'Ac': 0.01229, 'Ib': 2.22e-4, 'Ab': 0.01229},
            'W360x122': {'Ic': 3.80e-4, 'Ac': 0.01549, 'Ib': 3.80e-4, 'Ab': 0.01549},
        }
    
    # =========================================================================
    #  VISUALIZATION FUNCTIONS
    # =========================================================================

    def plot_simple_mode_shape(self, nodes, Ns, Nbl, sample_id, title_suffix=""):
        """Basic Wireframe Plot (Undeformed vs Deformed)"""
        plt.figure(figsize=(6, 6))
        # Plot Undeformed (Grey)
        for story in range(Ns):
            for bay in range(Nbl + 1):
                n1 = nodes.get((story, bay, 'struct'), nodes.get((story, bay)))
                n2 = nodes.get((story+1, bay))
                if n1 and n2:
                    plt.plot([ops.nodeCoord(n1)[0], ops.nodeCoord(n2)[0]], 
                             [ops.nodeCoord(n1)[1], ops.nodeCoord(n2)[1]], color='#BBBBBB', linestyle='-', linewidth=1.0)
        for story in range(1, Ns + 1):
            for bay in range(Nbl):
                n1, n2 = nodes.get((story, bay)), nodes.get((story, bay+1))
                if n1 and n2:
                    plt.plot([ops.nodeCoord(n1)[0], ops.nodeCoord(n2)[0]], 
                             [ops.nodeCoord(n1)[1], ops.nodeCoord(n2)[1]], color='#BBBBBB', linestyle='-', linewidth=1.0)
        
        # Plot Deformed (Red)
        scale = 5.0
        mode = 1
        try:
            for story in range(Ns):
                for bay in range(Nbl + 1):
                    n1 = nodes.get((story, bay, 'struct'), nodes.get((story, bay)))
                    n2 = nodes.get((story+1, bay))
                    if n1 and n2:
                        d1x, d1y = ops.nodeEigenvector(n1, mode, 1)*scale, ops.nodeEigenvector(n1, mode, 2)*scale
                        d2x, d2y = ops.nodeEigenvector(n2, mode, 1)*scale, ops.nodeEigenvector(n2, mode, 2)*scale
                        plt.plot([ops.nodeCoord(n1)[0]+d1x, ops.nodeCoord(n2)[0]+d2x], 
                                 [ops.nodeCoord(n1)[1]+d1y, ops.nodeCoord(n2)[1]+d2y], color='#D43F3A', linewidth=2.0)
            for story in range(1, Ns + 1):
                for bay in range(Nbl):
                    n1, n2 = nodes.get((story, bay)), nodes.get((story, bay+1))
                    if n1 and n2:
                        d1x, d1y = ops.nodeEigenvector(n1, mode, 1)*scale, ops.nodeEigenvector(n1, mode, 2)*scale
                        d2x, d2y = ops.nodeEigenvector(n2, mode, 1)*scale, ops.nodeEigenvector(n2, mode, 2)*scale
                        plt.plot([ops.nodeCoord(n1)[0]+d1x, ops.nodeCoord(n2)[0]+d2x], 
                                 [ops.nodeCoord(n1)[1]+d1y, ops.nodeCoord(n2)[1]+d2y], color='#D43F3A', linewidth=2.0)
        except: pass
        plt.title(f"Sample {sample_id}: Mode 1 (Deformed)\n{title_suffix}", pad=15)
        plt.xlabel("Distance (m)")
        plt.ylabel("Height (m)")
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"Sample_ModeShape_{sample_id}.png", bbox_inches='tight')
        plt.close()

    def plot_comparative_mode_shape(self, nodes_ssi, nodes_fb, Ns, Nbl, sample_id, title_suffix=""):
        """Comparison: Fixed Base (Blue Dashed) vs SSI (Red Solid)"""
        plt.figure(figsize=(8, 8))
        def plot_frame(nodes, color, linestyle, label=None, scale=1.0):
            mode = 1
            for story in range(Ns + 1):
                for bay in range(Nbl + 1):
                    if story < Ns: # Columns
                        n1 = nodes.get((story, bay, 'struct'), nodes.get((story, bay)))
                        n2 = nodes.get((story+1, bay))
                        if n1 and n2:
                            d1x, d1y = ops.nodeEigenvector(n1, mode, 1)*scale, ops.nodeEigenvector(n1, mode, 2)*scale
                            d2x, d2y = ops.nodeEigenvector(n2, mode, 1)*scale, ops.nodeEigenvector(n2, mode, 2)*scale
                            plt.plot([ops.nodeCoord(n1)[0]+d1x, ops.nodeCoord(n2)[0]+d2x], 
                                     [ops.nodeCoord(n1)[1]+d1y, ops.nodeCoord(n2)[1]+d2y], 
                                     color=color, linestyle=linestyle, linewidth=1.8, label=label if story==0 and bay==0 else "")
                    if story > 0 and bay < Nbl: # Beams
                        n1, n2 = nodes.get((story, bay)), nodes.get((story, bay+1))
                        if n1 and n2:
                            d1x, d1y = ops.nodeEigenvector(n1, mode, 1)*scale, ops.nodeEigenvector(n1, mode, 2)*scale
                            d2x, d2y = ops.nodeEigenvector(n2, mode, 1)*scale, ops.nodeEigenvector(n2, mode, 2)*scale
                            plt.plot([ops.nodeCoord(n1)[0]+d1x, ops.nodeCoord(n2)[0]+d2x], 
                                     [ops.nodeCoord(n1)[1]+d1y, ops.nodeCoord(n2)[1]+d2y], 
                                     color=color, linestyle=linestyle, linewidth=1.8)
        try:
            plot_frame(nodes_fb, color='#1f77b4', linestyle='--', label='Fixed Base', scale=5.0)
            plot_frame(nodes_ssi, color='#d62728', linestyle='-', label='SSI (Flexible Base)', scale=5.0)
            plt.title(f"Sample {sample_id}: Mode Shape Comparison\n{title_suffix}", pad=15)
            plt.xlabel("Distance (m)")
            plt.ylabel("Height (m)")
            plt.legend(frameon=True, fancybox=False, framealpha=0.9)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"Sample_Comparison_Mode_{sample_id}.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e: print(f"Plot error: {e}")

    # =========================================================================
    #  STATISTICAL SUITE FUNCTIONS
    # =========================================================================

    def plot_correlation_heatmap(self, df):
        cols = ['Vs', 'H', 'Ns', 'Nbl', 'L', 'rho_s', 'a0_dimensionless', 'T1_ratio', 'T1_SSI']
        data = df[cols].corr()
        plt.figure(figsize=(10, 8))
        im = plt.imshow(data, cmap='RdBu_r', vmin=-1, vmax=1)
        cbar = plt.colorbar(im, label='Pearson Correlation Coefficient')
        plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
        plt.yticks(range(len(cols)), cols)
        
        # Add values
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = data.iloc[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
                
        plt.title("Correlation Matrix of Input Parameters vs. Response")
        plt.tight_layout()
        plt.savefig("Analysis_Correlation_Matrix.png", bbox_inches='tight')
        plt.close()

    def generate_statistical_report(self, df):
        print("Generating report...")
        with open(self.stats_report_file, "w") as f:
            f.write("SSI FRAME ANALYSIS - STATISTICAL REPORT\n" + "="*60 + "\n\n")
            f.write(f"Total Samples: {len(df)}\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            
            f.write("DESCRIPTIVE STATISTICS\n" + "-"*40 + "\n")
            targets = ['T1_SSI', 'T1_fixed', 'T1_ratio', 'a0_dimensionless']
            f.write(df[targets].describe().to_string() + "\n\n")
            
            f.write("SKEWNESS & KURTOSIS\n" + "-"*40 + "\n")
            f.write(f"{'Variable':<20} | {'Skew':<10} | {'Kurtosis':<10}\n")
            for col in targets:
                f.write(f"{col:<20} | {df[col].skew():10.4f} | {df[col].kurt():10.4f}\n")
            f.write("\n")
            
            f.write("CORRELATION RANKING (vs T1_ratio)\n" + "-"*40 + "\n")
            f.write(df.corr()['T1_ratio'].sort_values(ascending=False).to_string() + "\n\n")
        print(f"Report saved to {self.stats_report_file}")

    def plot_sensitivity_boxplots(self, df):
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        df.boxplot(column='T1_ratio', by='Vs', ax=plt.gca(), grid=False, 
                   boxprops=dict(linewidth=1.2), medianprops=dict(color='red', linewidth=1.5))
        plt.title('Period Elongation vs. Shear Wave Velocity ($V_s$)')
        plt.xlabel('Shear Wave Velocity $V_s$ (m/s)')
        plt.ylabel('Period Ratio ($T_{SSI}/T_{Fixed}$)')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.suptitle('')
        
        plt.subplot(1, 2, 2)
        df.boxplot(column='T1_ratio', by='Ns', ax=plt.gca(), grid=False,
                   boxprops=dict(linewidth=1.2), medianprops=dict(color='red', linewidth=1.5))
        plt.title('Period Elongation vs. Number of Stories')
        plt.xlabel('Number of Stories ($N_s$)')
        plt.ylabel('Period Ratio ($T_{SSI}/T_{Fixed}$)')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig("Analysis_Sensitivity_Boxplots.png", bbox_inches='tight')
        plt.close()

    def plot_convergence(self, df):
        running_mean = df['T1_ratio'].expanding().mean()
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(df)), running_mean, color='#2ca02c', linewidth=1.5)
        plt.title('Convergence Check: Running Mean of Period Ratio')
        plt.xlabel('Number of Samples')
        plt.ylabel('Mean Period Ratio ($T_{SSI}/T_{Fixed}$)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("Analysis_Convergence_Check.png", bbox_inches='tight')
        plt.close()

    def plot_distribution_overlay(self, df):
        plt.figure(figsize=(10, 6))
        plt.hist(df['T1_fixed'], bins=50, alpha=0.6, label='Fixed Base', density=True, 
                 color='#1f77b4', edgecolor='black', linewidth=0.5)
        plt.hist(df['T1_SSI'], bins=50, alpha=0.6, label='SSI', density=True, 
                 color='#d62728', edgecolor='black', linewidth=0.5)
        plt.title('Probability Density of Fundamental Periods')
        plt.xlabel('Fundamental Period $T_1$ (s)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig("Analysis_Distribution_Overlay.png", bbox_inches='tight')
        plt.close()

    def plot_3d_interaction(self, df):
        try:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            p = ax.scatter(df['Vs'], df['Global_Aspect_Ratio'], df['T1_ratio'], 
                           c=df['T1_ratio'], cmap='viridis', s=15, alpha=0.7, edgecolors='none')
            ax.set_xlabel('\nShear Wave Velocity $V_s$ (m/s)', linespacing=2.0)
            ax.set_ylabel('\nAspect Ratio (H/W)', linespacing=2.0)
            ax.set_zlabel('\nPeriod Ratio', linespacing=2.0)
            ax.set_title('3D Interaction: Soil Stiffness vs. Structural Slenderness')
            ax.view_init(elev=25, azim=135)
            cbar = fig.colorbar(p, pad=0.1, shrink=0.7)
            cbar.set_label('Period Ratio')
            plt.tight_layout()
            plt.savefig("Analysis_3D_Interaction.png", bbox_inches='tight')
            plt.close()
        except: pass

    def plot_ecdf(self, df):
        try:
            data = np.sort(df['T1_ratio'])
            y = np.arange(1, len(data) + 1) / len(data)
            plt.figure(figsize=(10, 6))
            plt.plot(data, y, linestyle='-', color='#555555', linewidth=2.0)
            plt.fill_between(data, y, color='#E0E0E0', alpha=0.5)
            plt.title('Empirical Cumulative Distribution Function (ECDF)')
            plt.xlabel('Period Ratio ($T_{SSI}/T_{Fixed}$)')
            plt.ylabel('Cumulative Probability')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.savefig("Analysis_Probabilistic_ECDF.png", bbox_inches='tight')
            plt.close()
        except: pass

    def plot_veletsos_meek(self, df):
        try:
            h_eff = 0.7 * df['total_height']
            sigma = h_eff / (df['Vs'] * df['T1_fixed'].replace(0, 1e-10))
            plt.figure(figsize=(10, 6))
            sc = plt.scatter(sigma, df['T1_ratio'], alpha=0.5, c=df['Global_Aspect_Ratio'], 
                             cmap='plasma', s=20, edgecolors='none')
            cbar = plt.colorbar(sc, label='Global Aspect Ratio (H/W)')
            plt.xlabel(r'Relative Stiffness Parameter $\sigma = h_{eff} / (V_s \cdot T_{fixed})$')
            plt.ylabel(r'Period Elongation Ratio $\tilde{T}/T$')
            plt.title('Veletsos-Meek SSI Trend Verification')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig("Analysis_Veletsos_Trend.png", bbox_inches='tight')
            plt.close()
        except: pass

    def plot_hexbin_density(self, df):
        try:
            plt.figure(figsize=(10, 8))
            hb = plt.hexbin(df['Vs'], df['T1_ratio'], gridsize=40, cmap='magma_r', mincnt=1, bins='log')
            cbar = plt.colorbar(hb, label='Log Count ($N$)')
            plt.xlabel('Shear Wave Velocity $V_s$ (m/s)')
            plt.ylabel('Period Ratio ($T_{SSI}/T_{Fixed}$)')
            plt.title('Density of Period Elongation vs. Soil Stiffness')
            plt.tight_layout()
            plt.savefig("Analysis_Hexbin_Density.png", bbox_inches='tight')
            plt.close()
        except: pass

    def plot_violin_distribution(self, df):
        try:
            plt.figure(figsize=(12, 6))
            data = [df[df['Ns'] == n]['T1_ratio'].values for n in sorted(df['Ns'].unique())]
            parts = plt.violinplot(data, showmeans=False, showmedians=True)
            
            # Style violins
            for pc in parts['bodies']:
                pc.set_facecolor('#8DA0CB')
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            parts['cmedians'].set_color('#D95F02')
            
            plt.xlabel('Number of Stories')
            plt.ylabel('Period Ratio ($T_{SSI}/T_{Fixed}$)')
            plt.xticks(range(1, len(data) + 1), sorted(df['Ns'].unique()))
            plt.title('Distribution of SSI Effects by Building Height')
            plt.grid(axis='y', linestyle=':', alpha=0.5)
            plt.tight_layout()
            plt.savefig("Analysis_Violin_Distribution.png", bbox_inches='tight')
            plt.close()
        except: pass

    def plot_contour_interaction(self, df):
        try:
            plt.figure(figsize=(10, 8))
            triang = tri.Triangulation(df['Vs'], df['Global_Aspect_Ratio'])
            
            # Filled contours
            cntr = plt.tricontourf(triang, df['T1_ratio'], levels=15, cmap='viridis')
            cbar = plt.colorbar(cntr, label='Period Ratio')
            
            # Line contours
            line_cntr = plt.tricontour(triang, df['T1_ratio'], levels=15, colors='white', linewidths=0.5, alpha=0.5)
            
            plt.xlabel('Shear Wave Velocity $V_s$ (m/s)')
            plt.ylabel('Global Aspect Ratio (H/W)')
            plt.title('SSI Interaction Contour Map')
            plt.tight_layout()
            plt.savefig("Analysis_Contour_Interaction.png", bbox_inches='tight')
            plt.close()
        except: pass

    def plot_dataset_summary(self, df):
        plt.figure(figsize=(16, 10))
        
        # Subplot 1
        plt.subplot(2, 2, 1)
        plt.hist(df['T1_SSI'], bins=50, color='#66C2A5', edgecolor='black', linewidth=0.5)
        plt.title('(a) Distribution of SSI Fundamental Periods')
        plt.xlabel('Period $T_1$ (s)')
        plt.ylabel('Frequency')
        
        # Subplot 2
        plt.subplot(2, 2, 2)
        plt.scatter(df['T1_fixed'], df['T1_SSI'], alpha=0.4, s=8, color='#8DA0CB', edgecolors='none')
        plt.plot([0, df['T1_fixed'].max()], [0, df['T1_fixed'].max()], 'r--', linewidth=1.5, label='1:1 Line')
        plt.title('(b) SSI vs. Fixed-Base Period')
        plt.xlabel('Fixed Base Period (s)')
        plt.ylabel('SSI Period (s)')
        plt.legend()
        
        # Subplot 3
        plt.subplot(2, 2, 3)
        plt.scatter(df['Global_Aspect_Ratio'], df['T1_ratio'], c=df['Vs'], cmap='viridis', s=8, alpha=0.6, edgecolors='none')
        cbar = plt.colorbar(label='$V_s$ (m/s)')
        plt.title('(c) Slenderness vs. Elongation')
        plt.xlabel('Aspect Ratio (H/W)')
        plt.ylabel('Period Ratio')
        
        # Subplot 4
        plt.subplot(2, 2, 4)
        plt.scatter(df['a0_dimensionless'], df['T1_ratio'], color='#FC8D62', s=8, alpha=0.4, edgecolors='none')
        plt.title('(d) Dimensionless Frequency ($a_0$) vs. Ratio')
        plt.xlabel('Dimensionless Frequency $a_0$')
        plt.ylabel('Period Ratio')
        
        plt.tight_layout()
        plt.savefig("Analysis_Dashboard_General.png", bbox_inches='tight')
        plt.close()

    def perform_full_statistical_suite(self):
        if not self.output_file or not os.path.exists(self.output_file):
            print("No output file found.")
            return

        print("\n" + "="*50 + "\nSTARTING FULL STATISTICAL ANALYSIS\n" + "="*50)
        try:
            df = pd.read_csv(self.output_file)
            
            # Call ALL Plotting Functions
            self.generate_statistical_report(df)
            self.plot_correlation_heatmap(df)
            self.plot_sensitivity_boxplots(df)
            self.plot_convergence(df)
            self.plot_distribution_overlay(df)
            self.plot_3d_interaction(df)
            self.plot_ecdf(df)
            self.plot_veletsos_meek(df)
            self.plot_hexbin_density(df)
            self.plot_violin_distribution(df)
            self.plot_contour_interaction(df)
            self.plot_dataset_summary(df)
            
            print("Analysis Complete. Check generated PNG and TXT files.")
        except Exception as e:
            print(f"Error during stats analysis: {e}")

    def save_results_chunk(self, results_chunk, mode='a'):
        if not results_chunk: return
        df_chunk = pd.DataFrame(results_chunk)
        write_header = not self.header_written
        df_chunk.to_csv(self.output_file, mode=mode, header=write_header, index=False)
        if write_header: self.header_written = True
        del df_chunk
        del results_chunk[:]

    def run_parametric_study(self):
        print("Generating parameter combinations...")
        
        # Params
        Vs_range = np.arange(100, 801, 200)
        H_range = np.arange(3.0, 4.01, 0.5)
        Ec_range = [20000, 30000, 40000]
        MAX_STORIES = 10 
        Ns_range = range(1, MAX_STORIES + 1)
        Nbl_range = range(1, 11)
        L_range = np.arange(5.0, 7.01, 1.0)
        rho_s_range = np.arange(16.0, 25.01, 1.0)
        footing_range = np.arange(1.5, 2.01, 0.5)
        
        steel_sections = self.get_steel_sections()
        section_names = list(steel_sections.keys())
        base_params = {'rho_c': 77.0, 'nu_s': 0.3}
        
        all_combinations = list(itertools.product(
            Vs_range, H_range, Ec_range, section_names, 
            Ns_range, Nbl_range, L_range, rho_s_range, footing_range
        ))
        
        target_samples = 300000
        selected_combinations = random.sample(all_combinations, target_samples) if len(all_combinations) > target_samples else all_combinations
        total_to_run = len(selected_combinations)
        
        self.output_file = f'SSI_Steel_Frame_ML_Dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        print(f"Total to run: {total_to_run:,} | Output: {self.output_file}")
        
        results = []
        count = 0
        error_count = 0
        start_time = datetime.now()
        chunk_size = 5000
        
        for params_tuple in selected_combinations:
            (Vs, H, Ec, section_name, Ns, Nbl, L, rho_s, footing_dim) = params_tuple
            section_props = steel_sections[section_name]
            
            params = base_params.copy()
            params.update({
                'Vs': Vs, 'H': H, 'Ec': Ec, 'Ic': section_props['Ic'], 'Ac': section_props['Ac'],
                'Ib': section_props['Ib'], 'Ab': section_props['Ab'], 'Ns': Ns, 'Nbl': Nbl,
                'L': L, 'rho_s': rho_s, 'footing_dim': [footing_dim, footing_dim]
            })
            
            # Fixed Base
            params['include_ssi'] = False
            try:
                nodes_fb = self.create_2D_frame_model(params)
                per_fb, fr_fb = self.perform_eigenvalue_analysis(5)
                T1_fb, T2_fb, T3_fb, f1_fb = per_fb[0], per_fb[1], per_fb[2], fr_fb[0]
            except Exception as e:
                T1_fb = T2_fb = T3_fb = f1_fb = 0.0
                error_count += 1
                self.log_error(params, f"Fixed Base Error: {e}")

            # SSI
            params['include_ssi'] = True
            try:
                nodes_ssi = self.create_2D_frame_model(params)
                per_ssi, fr_ssi = self.perform_eigenvalue_analysis(5)
                T1_ssi, T2_ssi, T3_ssi, f1_ssi = per_ssi[0], per_ssi[1], per_ssi[2], fr_ssi[0]
            except Exception as e:
                T1_ssi = T2_ssi = T3_ssi = f1_ssi = 0.0
                error_count += 1
                self.log_error(params, f"SSI Error: {e}")
            
            Gs = rho_s * Vs**2
            Kv, Kh, Kr = self.calculate_spring_stiffness(Gs, base_params['nu_s'], footing_dim*footing_dim, footing_dim, footing_dim)
            
            # Dimensionless
            r_equiv = sqrt(footing_dim*footing_dim / pi) 
            omega_ssi = 2 * pi * f1_ssi if f1_ssi > 0 else 0
            a0 = (omega_ssi * r_equiv) / Vs if Vs > 0 else 0
            
            # Mode Shapes
            mode_shapes = self.get_mode_shapes(nodes_ssi if 'nodes_ssi' in locals() else {}, Ns, MAX_STORIES, 3)
            
            # PER-SAMPLE VISUALIZATION (Limit to first 5)
            if count < 5 and T1_ssi > 0 and 'nodes_ssi' in locals():
                self.plot_simple_mode_shape(nodes_ssi, Ns, Nbl, count+1, f"Vs={Vs}, Stories={Ns}")
                if 'nodes_fb' in locals():
                    self.plot_comparative_mode_shape(nodes_ssi, nodes_fb, Ns, Nbl, count+1, f"Vs={Vs}")

            # Store Result
            result = {
                'Vs': Vs, 'H': H, 'Ec': Ec, 'Ic': section_props['Ic'], 'Ac': section_props['Ac'],
                'Ns': Ns, 'Nbl': Nbl, 'L': L, 'rho_s': rho_s, 'footing_dim': footing_dim,
                'Gs': Gs, 'Kv': Kv, 'Kh': Kh, 'Kr': Kr, 'a0_dimensionless': a0,
                'total_height': Ns * H, 'total_width': Nbl * L,
                'Story_Aspect_Ratio': H/L, 'Global_Aspect_Ratio': (Ns*H)/(Nbl*L),
                'T1_fixed': T1_fb, 'T2_fixed': T2_fb, 'T3_fixed': T3_fb, 'f1_fixed': f1_fb,
                'T1_SSI': T1_ssi, 'T2_SSI': T2_ssi, 'T3_SSI': T3_ssi, 'f1_SSI': f1_ssi,
                'T1_ratio': T1_ssi/T1_fb if T1_fb>0 else 0,
                'T2_ratio': T2_ssi/T2_fb if T2_fb>0 else 0,
                'T3_ratio': T3_ssi/T3_fb if T3_fb>0 else 0,
                'f1_ratio': f1_ssi/f1_fb if f1_fb>0 else 0,
            }
            result.update(mode_shapes)
            results.append(result)
            count += 1
            
            if count % chunk_size == 0:
                self.save_results_chunk(results)
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = count / elapsed if elapsed > 0 else 0
                print(f"Progress: {count:,}/{total_to_run:,} | Rate: {rate:.1f}/s | Errors: {error_count}")
        
        if len(results) > 0: self.save_results_chunk(results)
        
        # FULL STATS AT END
        self.perform_full_statistical_suite()
        return count

if __name__ == "__main__":
    print("\n" + "="*70 + "\nSSI PARAMETRIC STUDY - ULTIMATE EDITION\n" + "="*70)
    analyzer = SSI_FrameAnalysis()
    total = analyzer.run_parametric_study()
    print(f"\nDONE. Total: {total:,}. Output: {analyzer.output_file}")