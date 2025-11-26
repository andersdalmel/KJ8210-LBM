import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from tqdm import tqdm
import FD


class LBM():
    '''
    Class to implement the D2Q9 Lattice Boltzmann Model.
    Must contain:
        - A constructor to initialise an NxM lattice and particles on it, 
            as well as relevant quantities
        - A function to implement the collision step
        - A function to implement the propagation step
        - A function to calculate equilibrium distribution
        - Functions to do the same for the concentration field
        - After a while: a function to visualise, i.e. animate the simulations
    '''

    def __init__(self, Nx, Ny, nu, L, D, debug = False):
        '''
        N = number of lattice points per row
        M = number of lattice points per col
        nu = viscosity
        w = omega, relaxation time, 0 < w < 2.
        L = system length (square lattice?)
        D = diffusion constant

        c_i   : (9, 2)
        Q_iab : (9, 2, 2)
        N     : (9, Nx, Ny)
        C     : (9, Nx, Ny)
        '''
        self.Nx = Nx
        self.Ny = Ny
        self.nu = nu
        self.w = 1/(3*nu+0.5)
        self.L = L
        self.D = D
        self.w_D = 1/(3*D+0.5)
        x = np.linspace(0, L, Nx)
        y = np.linspace(0, L, Ny)
        self.xx, self.yy = np.meshgrid(x, y, indexing="ij") # xx[i,j] is the x coord. of lattice site (i,j)
                                                            # yy[i,j] is likewise for the y coord.
        self.weights = np.array([4/9, 1/9, 1/9, 1/9, 
                                 1/9, 1/36, 1/36, 1/36, 
                                 1/36], dtype = np.float64)
        self.c_i_int = np.array([[0,0], [1,0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], 
                             [-1, -1], [1, -1]], dtype = np.int32) # for propagate() func
        self.c_i_float = self.c_i_int.astype(np.float64)           # for arithmetic
        self.kronecker_delta = np.eye(2, dtype = np.int32)
        self.Q_iab = 1.5*np.einsum("ia,ib -> iab", self.c_i_float, self.c_i_float) - 0.5*self.kronecker_delta
        self.rho = np.ones((Nx, Ny)) # initialise to uniform density
        self.mass = np.sum(self.rho)
        self.u = self.__u_init(self.xx, self.yy)
        if debug: 
            self.__u_dbg()
        self.N = self.__N_init()
        self.C = self.__C_init(self.xx, self.yy)  # (9, Nx, Ny)
        self.C_tot = np.sum(np.sum(self.C, axis = 0))
        if debug:
            self.__C_dbg()
        self.p_x = np.sum(self.rho*self.u[:, :, 0])
        self.p_y = np.sum(self.rho*self.u[:, :, 1])
        self.t = 0
        diff = np.abs(np.sum(self.C, axis = 0)[:, self.Ny//2:] - self.C_0 / 2)
        y_midIndex = np.argmin(diff, axis = 1)
        y_mid0 = np.zeros(self.Nx)
        for i in range(self.Nx):
            y_mid0[i] = self.yy[i, (self.Ny//2) + y_midIndex[i]]
        self.y_mid0 = y_mid0
        self.etas = []
        self.dSdt = []

    def __u_init(self, x, y):
        '''
        Function initialise the velocity field according to Eq.s (1), (2).
        '''
        U = 0.1
        d = 1
        u_0y = 0.001
        k = 2*np.pi/(self.L/10)
        u_x = U*(np.tanh((y-0.25*self.L)/d) - np.tanh((y-0.75*self.L)/d) - 1)
        # u_x = U*np.heaviside(y-self.L/2, 0.5) # for fun
        u_y = np.heaviside(x-self.L, 0)# u_0y*np.sin(k*x) #task 3 check
        return np.stack([u_x, u_y], axis = -1, dtype = np.float64)
    
    def __u_dbg(self):
        self.u[:, :, 0] = 1
        self.u[:, :, 1] = 0
        
    def __C_dbg(self):
        self.C[:, :, :] = 0
        self.C[1, 10:25, 10:25] = 1
     
    def __C_init(self, x, y):
        '''
        Function to initialise concentration field according to Eq. (3). 
        Func. requires both x and y coordinates for consistency though only y coords are used.
        '''
        C_0 = 1.0
        self.C_0 = C_0
        d = 1

        u_0y = 0.001
        k = 2*np.pi/(self.L/10)
    
        C = (C_0/2)*(np.tanh((y-0.25*self.L)/d) - np.tanh((y-0.75*self.L)/d))*u_0y*np.sin(k*x)
        # C = np.heaviside(y-self.L/2, 0.5) #for fun
        C_init = self.weights[:, None, None]*C[None, :, :]*(1+3*np.einsum("ia, xya -> ixy", self.c_i_float, self.u))

        return C_init
        
    
    def __N_init(self):
        '''
        Shapes: 
            weights : (9)
            rho     : (Nx, Ny)
            c_i     : (9, 2)
            u       : (Nx, Ny, 2)
        Want shape: 
            N_eq    : (9, Nx, Ny)
        '''
        N_eq = self.weights[:, None, None]*self.rho[None, :, :]*(1+3*np.einsum("ia, xya -> ixy", self.c_i_float, self.u) 
                                    + 3*np.einsum("iab, xya, xyb -> ixy", self.Q_iab, self.u, self.u))
        
        return N_eq
    
    

    # def getRho(self, pos):
    #     'Obsobs: using this will be way too slow to be feasible. Make the calculations inline & vectorise.'
    #     x, y = pos
    #     rho = np.sum(self.N[:, x, y])
    #     return rho

    # def getU(self, pos, rho):
    #     'Obsobs: using this will be way too slow to be feasible. Make the calculations inline & vectorise.'
    #     x, y = pos
    #     u = (np.sum(self.N[:,x,y]*self.c_i, axis = 0))/rho
    #     return u

    
    # def getC(self, pos):
    #     'Obsobs: using this will be way too slow to be feasible. Make the calculations inline & vectorise.'
    #     x, y = pos
    #     mass = np.sum(self.C[:, x, y])
    #     return mass


    def collision_N(self, debug):
        'All calculations should be inline and vectorised. Avoid loops --> treat the entire lattice.'
        self.rho = np.sum(self.N, axis = 0)
        self.u = np.einsum("ixy, ia -> xya", self.N, self.c_i_float) / self.rho[:, :, None]
        # cu = np.einsum("ia, xya -> ixy", self.c_i_float, self.u) #DEBUG
        # u2 = np.einsum("xya, xya -> xy", self.u, self.u)

        self.N_eq = self.weights[:, None, None]*self.rho[None, :, :]*(1+3*np.einsum("ia, xya -> ixy", self.c_i_float, self.u) 
                                   + 3*np.einsum("iab, xya, xyb -> ixy", self.Q_iab, self.u, self.u))

        # self.N_eq = self.weights[:, None, None]*self.rho[None, :, :]*(1+3*np.einsum("ia, xya -> ixy", self.c_i_float, self.u)
        #                             + 4.5*cu**2 -1.5*u2) #check if tensor implement is the problem. it is not. 

        if debug:
            ###DEBUG##
            # after computing self.N_eq but before forming N_coll
            rho_recov = np.sum(self.N_eq, axis=0)      # shape (Nx, Ny)
            err = rho_recov - self.rho                # per-site error

            print("rho_recov sum-diff:", np.sum(err))            # should be ~0
            print("rho_recov max abs:", np.max(np.abs(err)))     # should be ~1e-14..1e-12
            print("rho_recov stats:", np.min(err), np.mean(err), np.max(err))
            # if you want the location of worst site:
            imax, jmax = np.unravel_index(np.argmax(np.abs(err)), err.shape)
            print("worst site error:", imax, jmax, err[imax, jmax])

            # also check momentum recovery from N_eq:
            mom_recov = np.einsum("ixy,ia->xya", self.N_eq, self.c_i_float)  # (Nx,Ny,2)
            mom_err = mom_recov - (self.rho[:, :, None] * self.u)
            print("momentum max abs:", np.max(np.abs(mom_err)))
            print()
            ############


        N_coll = self.N - self.w*(self.N - self.N_eq)
        return N_coll

    def collision_C(self):
        C = np.sum(self.C, axis = 0)
        self.C_eq = self.weights[:, None, None]*C[None, :, :]*(1+3*np.einsum("ia, xya -> ixy", self.c_i_float, self.u))
        C_coll = self.C - self.w_D*(self.C - self.C_eq)
        return C_coll

    def propagate_N(self, N_coll):
        '''
        This takes in N_i(x, t) and must propagate it to N_i(x+c_i, t+1). 
        Shape: N_coll  -  (9, Nx, Ny).
        OBS: must implement periodic boundary conditions. 
        '''
        
        for i in range(9):
            self.N[i] = np.roll(N_coll[i], self.c_i_int[i], axis = (0,1))


    def propagate_C(self, C_coll):
        for i in range(9):
            self.C[i] = np.roll(C_coll[i], self.c_i_int[i], axis = (0,1))


    def findEta(self):
        '''
        There will be a y_mid(x) \in [L/2, L] which is such that the number |C(x,y) - C_0/2| is minimal. 
        This function aims to obtain y_mid(x, t) and y_mid(x, t=0) and plot how their difference increases with time. 
        Wait a minute, this likely means we need a separate y_mid for each x_val?? N_x y_mids per time step?
        Specifically, we wish to verify the \eta(x,t) = y_mid(x, t) - y_mid(x, t=0) \propto exp(kUt) relation. 
        We only need to investigate early time evolution, meaning we don't need as long simulations as earlier. 
        '''
        C_0 = self.C_0
        C_tot = (np.sum(self.C, axis = 0)) #shape (Nx, Ny)
        diff = np.abs(C_tot[:, self.Ny//2:] - C_0 / 2)
        minList = np.argmin(diff, axis = 1) #WE WANT THE Y-VALUE AT THE INDICES, NOT JUST INDICES
        y_mids = np.zeros(self.Nx)
        for i in range(self.Nx):
            y_mids[i] = self.yy[i, (self.Ny//2) + minList[i]]
        eta = y_mids - self.y_mid0 # Shape (Nx)
        self.etas.append(eta)

    def gradC(self):
        '''
        Function assumes it is run after C-propagation step, such that self.C_eq and self.C are  
        updated quantities. 
        '''
        diff = self.C - self.C_eq
        gradC = 3*self.w_D*np.einsum("ia, ixy -> xya", self.c_i_float, diff)
        return gradC

    def calcEntropyProductionRate(self):
        '''
        Function to calculate entropy production rate from statistical mechanics formula, 
        specifically calculates (1/Dk_B)dS/dt = \int gradC^2 / C dV. 
        Uses self.gradC() so assumes it is run after C propagation step. 
        Appends results to class variable list. 
        '''
        C_tot = np.sum(self.C, axis = 0) # (Nx, Ny)
        gradC = self.gradC() # (Nx, Ny, 2)
        gradCsquared = np.einsum("xya, xya -> xy", gradC, gradC)
        # gradC = np.gradient(C_tot)
        integrand = (gradCsquared)/C_tot
        sProdRate = np.sum(integrand)
        self.dSdt.append(sProdRate)


    def onlyNsim(self, stopTime, diagnostic = 0):
        while self.t < stopTime:

            if diagnostic: 
                M_before = np.sum(self.N)
                N_coll = self.collision_N(diagnostic)
                M_after_collision = np.sum(N_coll)
                self.propagate_N(N_coll)
                M_after_stream = np.sum(self.N)

                print(M_before, M_after_collision, M_after_stream)
            else: 
                N_coll = self.collision_N()
                self.propagate_N(N_coll)
            self.t += 1
        # print((np.sum(self.N, axis = 0) - np.sum(self.N_eq, axis = 0)).shape)


    def fullSim(self, stopTime, plot = [], save = True, interval = 500, diagnostic = False, 
                        findEta = 1, findSprod = 1):
        progBar = tqdm(total = stopTime, desc = "Running simulation")

        self.snaps_t = []
        self.snaps_C = []
        self.snaps_u = []

        while self.t < stopTime:
            N_coll = self.collision_N(False)
            C_coll = self.collision_C()
            self.propagate_N(N_coll)
            self.propagate_C(C_coll)
            if diagnostic and self.t > 10: 
                # print(f"Step: {self.t}")
                self.checkMassConservation()
                self.checkMomentumConservation()
                self.checkCConservation()
            self.t += 1
            if len(plot) > 0: 
                if self.t % interval == 0:
                    self.snaps_t.append(self.t)
                    for qt in plot:
                        if qt == "C":
                            C = np.sum(self.C, axis = 0)
                            self.snaps_C.append(C)
                        if qt == "u":
                            self.snaps_u.append(self.u)
            if findEta:
                self.findEta()
            if findSprod and self.t > 100:
                self.calcEntropyProductionRate()
    
            progBar.update(1)
        if len(plot) > 0:
            np.save("u_array", np.array(self.snaps_u))
            self.makeAni(plot, interval, save)
        if findEta:
            self.plotEta()
        if findSprod:
            self.plotdSdt()

    def plotEta(self):
        t_vals = np.arange(len(self.etas))
        print(np.array(self.etas).shape)
        eta_max = np.max(np.abs(np.array(self.etas)), axis = 1)
        plt.figure(figsize=(8, 4))
        plt.plot(t_vals, np.log(eta_max), marker=".")
        plt.xlabel("time steps")
        plt.ylabel("(max(|η|))")
        plt.title("Growth over time of a perturbation η")
        plt.savefig("Perturbation growth exp check.png", dpi = 1000)
        plt.show()

    def plotdSdt(self):
        t_vals = np.arange(100, len(self.dSdt)+100)
        dSdt = self.dSdt
        plt.figure(figsize=(8, 4))
        plt.plot(t_vals, dSdt)
        plt.xlabel("time steps")
        plt.ylabel(r"$(1/Dk_B) dS/dt$")
        plt.title("Entropy production rate")
        plt.savefig("SprodRate.png", dpi = 1000)
        plt.show()

    def makeAni(self, quantities, interval = 500, save = True):
        for qt in quantities:
            if qt == "C":
                snaps_C = self.snaps_C
                times = self.snaps_t

                fig, ax = plt.subplots(figsize = (6, 5))
                im = ax.imshow(
                    snaps_C[0].T, #transpose to get correct axes
                    origin = "lower",
                    cmap = "cividis",
                    extent = [0, self.L, 0, self.L],
                    vmin = 0,
                    vmax = 1
                )
                fig.colorbar(im, ax = ax)
                ax.set_title(f"t = {times[0]}")

                def update(i):
                    im.set_data(snaps_C[i].T)
                    ax.set_title(f"t = {times[i]}")
                    return [im]
                
                ani = FuncAnimation(fig, update, frames = len(snaps_C), interval = interval)
                if save:
                    ani.save("C_snaps.gif", fps = 24, dpi = 200)
                
                plt.show()
                plt.close()
            
            if qt == "u":
                snaps_u = self.snaps_u
                times = self.snaps_t

                fig, ax = plt.subplots(figsize = (6,6))
                ax.set_title(f"Velocity field u, t = {times[0]}")

                u_init = snaps_u[0]
                im = ax.quiver(self.xx, self.yy, u_init[:, :, 0], u_init[:, :, 1]) # potentially add scale 
                ax.set_xlim(0, self.Nx)
                ax.set_ylim(0, self.Ny)

                def update(i):
                    u = snaps_u[i]
                    im.set_UVC(u[:, :, 0], u[:, :, 1])
                    ax.set_title(f"Velocity field u, t = {times[i]}")

                ani = FuncAnimation(fig, update, frames = len(snaps_u), interval = interval)
                if save:
                    ani.save("u_snaps.gif", fps = 24, dpi = 200)

                plt.show()
                plt.close()



    ######## DIAGNOSTIC FUNCS #########

    def checkMassConservation(self):
        mass = np.sum(self.rho)
        assert np.abs(mass - self.mass) < 1e-8, f"Mass is not conserved. \ndelta_m = {mass - self.mass}, step: {self.t}"
        self.mass = mass

    def checkCConservation(self):
        C = np.sum(np.sum(self.C, axis = 0))
        assert np.abs(C - self.C_tot) < 1e-8, f"C is not conserved. \ndelta_C = {C-self.C_tot}, step: {self.t}"
        self.C_tot = C


    def checkMomentumConservation(self):
        p_x = np.sum(self.rho*self.u[:, :, 0])
        p_y = np.sum(self.rho*self.u[:, :, 1])
        assert np.abs(p_x - self.p_x) < 1e-8 and np.abs(p_y - self.p_y) < 1e-8, f"Momentum is not conserved\ndelta_px = {p_x - self.p_x}, delta_py = {p_y - self.p_y}, step: {self.t}"
        self.p_x = p_x
        self.p_y = p_y
    
    ####################################



if __name__ == "__main__":

    test1 = LBM(200, 200, 0.01, 200, 0.01, debug = 0) 
    # # test2 = LBM(32, 32, 0.01, 2, 200, 0.01) 

    startTime = time.time()
    interval = 250

    # test1.fullSim(1000, plot = False, interval = 100, diagnostic = True)
    test1.fullSim(50000, diagnostic = 1, plot = ["C"], interval = interval, findEta = 0)
    # test1.makeAni(["C"], interval = interval, save = 1)
    midTime = time.time()
    # # test2.noOverheadSim(2500)
    # # endTime = time.time()

    # print(f"Time consumed with function calls: {midTime - startTime} seconds")
    # # print(f"Time consumed without function calls: {endTime - midTime} seconds")
    # print(test1.N[0])
    # # print(test2.N[0])