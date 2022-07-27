import numpy as np
import datetime
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import cm
from math import trunc
import random as rnd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import inv
import pandas as pd

import NeQuick_2


# define main constants
R = 6371.  # Earth radius in km
R_sat = 7371.  # Radius of the satellites orbit in km
#Upper boundary of the grid must be equel heigth of the Sattelaite

# define geometry of scaning directions
# height, tay - coordinates in km
# betta - inclination of the scaning ray below satellite trajectory in degrees

directory = ('C:\\Users\\IZMIRAN1\\2003-10-13_COSMOS-2389\\200310131910v.rfm')
data1 = pd.read_table(directory, header = None, sep = '\s+', skiprows = 10)
data1.columns = ['Time', 'Lat', 'Phase', 'Lon', 'Ampl', 'El', 'Az', 'TEC']


data1 = data1[:100]
#data_TEC = data[['Lat', 'TEC']]
#data_TEC


#plt.plot(data.Lat, data.TEC)

class TRay:
   
   def __init__(self, tay_rec, height_sat, betta, azimuth):
       self.tay_rec = tay_rec
       self.height_sat = R_sat - R
       self.betta = betta
       self.azimuth = azimuth
       
       
   
  
   def tay_from_height(self, height):
       return self.tay_rec + R * (np.radians(self.betta) - np.arccos(R * 
                                np.cos(np.radians(self.betta)) / (R + height)))


# define grid parameters 
# height_up should be always greater than R_sat-R
#границы сетки выражаются в ~град*100 км.
class TGrid:
    
    '''
    Задание сетки для модели NeQuick2 и дальнейшего моделирования
    
    tau_left - левая граница сетки модели (5700 ~ 50 град с.ш.), 
    tau_right - привая граница сетки модели,
    height_down - нижняя граница сетки (0 соотвтетсвует поверхности Земли),
    height_up - верхняя граница сетки (=R_sat-R),
    tau_nodes - количество узлов сетки по горизонтале, 
    height_nodes - количество узлов сетки по вертикали
    '''
    
    def __init__(self, tay_left=5550., tay_right=8905.6, 
                 height_down=0., height_up=1000., tay_nodes=101, height_nodes=61):
        self.tay_left = tay_left
        self.tay_right = tay_right
        self.height_down = height_down
        self.height_up = height_up
        self.tay_nodes = tay_nodes
        self.height_nodes = height_nodes
        self.dtay = (self.tay_right - self.tay_left) / (self.tay_nodes - 1)
        self.dheight = (self.height_up - self.height_down) / (self.height_nodes - 1)


class Background:
    
    '''
    Задание фона - вычисление модели NeQuick2 и добавление к ней неоднородностей
    
    month - месяц; hour - час UT; f10p7 - F10.7; lon - долгота, град; 
    N_ir_max_arr - добавка к эл. концентрации от неоднородности;
    tay0_arr - координата центра неоднородности; h0_arr - высота центра неоднородности;
    a_arr и b_arr - размеры неоднородности; gamma_arr - угод наклона неоднородности; 
    v_arr - (?)
    '''
    
    def __init__(self, month=6, hour=12, f10p7=90, lon=20,
                 N_ir_max_arr=[0.5, ], h0_arr=[150.,],
                 tay0_arr=[6700.,], a_arr=[60., ],
                 b_arr=[5000.,], gamma_arr=[0., ], v_arr=[1., ]):
        
        '''
        Инициализация необходимых параметров (состояние ионосферы/атмосферы и 
                                              параметры неоднородностей)
        '''
        
        self.month = month
        self.hour = hour
        self.f10p7 = f10p7
        self.lon = lon
        self.N_ir_max_arr = N_ir_max_arr
        self.h0_arr = h0_arr
        self.tay0_arr = tay0_arr
        self.a_arr = a_arr
        self.b_arr = b_arr
        self.gamma_arr = gamma_arr
        self.v_arr = v_arr
    
    def zerostate(self, tay, height):
        
        '''
        Вычисление модели NeQuick2. Возвращает N-units
        '''
        
        ne = NeQuick_2.nequick(height, np.rad2deg(tay/R), self.lon, 
                               self.month, self.f10p7, self.hour)

        if ne < 0 or np.isnan(ne):
            ne = 0.
        return ne * 1e-12
    
    def irregularity(self, tay, height, N_ir_max, h0, tay0, a, b, gamma, v):
        
        '''
        Вычисление неоднородностей.
        '''
        
        alpha = ((height - h0) * np.cos(np.radians(gamma)) - 
                 (tay - tay0) * np.sin(np.radians(gamma)))**2. / a**2. + \
                ((height - h0) * np.sin(np.radians(gamma)) + 
                 (tay - tay0) * np.cos(np.radians(gamma)))**2. / b**2.
        if alpha <= 1.:
            result = N_ir_max * (np.cos(np.pi*np.sqrt(alpha) / 2.))**2.
        else: 
            result = 0.
        return result
    
    def model(self, tay, height):
        
        """
        Добавлене неоднородностей к NeQuick2. Возвращает N-units.
        """
        
        result = self.zerostate(tay, height)
        for N_ir_max, h0, tay0, a, b, gamma, v in zip(self.N_ir_max_arr, 
            self.h0_arr, self.tay0_arr, self.a_arr, self.b_arr, self.gamma_arr, self.v_arr):
            if v == 1.:
                result = result + self.irregularity(tay, height, N_ir_max, h0, tay0, a, b, gamma, v)
            if v == 0.:
                result = result - self.zerostate(tay, height) * self.irregularity(
                    tay, height, N_ir_max, h0, tay0, a, b, gamma, v)
        if result <= 0.:
            result = 0.
        return result  # returns eldens in N-unit

####################################################################################


def CreateRays(rec_tays, Grid, data1):
    
    """
    Создание лучей просвечивания приёмников.
    """

    Rays = []

    for rec_tay in rec_tays:
        for b in data1.El:  # Ширина лучей приёмника, было arange(5, 180, 0.5)
            for a in data1.Az:
                if(
                TRay(rec_tay, R_sat - R, b,a).tay_from_height(Grid.height_up) < Grid.tay_right
                and TRay(rec_tay, R_sat - R, b,a).tay_from_height(Grid.height_up) > Grid.tay_left
                    ):
                    Rays.append(TRay(rec_tay, R_sat - R, b,a))
    '''
    условие if не даёт лучам выйти за пределы сформированной сетки
    '''
    return Rays


def ComputeRHS(Rays, Model):
    
    """
    Генерация псевдонаблюдений, вычисление ПЭС. Возвращает TECu
    """
    
    RHS = []

    def f1(height, Ray, Model): # Интеграл для вычисления ПЭС с учётом Якобиана преобразования
        return Model.model(Ray.tay_from_height(height), height) \
            * (R + height) / (np.sqrt(height**2 + 2.*R*height \
            + R**2 * (np.sin(np.deg2rad(Ray.betta)))**2))
                
    def Noise(betta, sigma0= 0.05):  # аддитивный шум, зависит от синуса угла возвышения спутника
        return rnd.gauss(0, sigma0/np.sin(np.deg2rad(betta)))
        

    i = 0
    for Ray in Rays:
        #print(i)
        i = i + 1
        I = quad(f1, 0, Ray.height_sat, args=(Ray, Model), epsabs=1e-2)
        #print(Noise(Ray.betta))
        RHS.append(I[0] + Noise(Ray.betta))

    return np.array(RHS)/10. # return in TECu

def ObsOperator(Grid, Rays):
    
    """
    Оператор наблюдения. Служит для работы фильтра Калмана
    """
    
    tnodes = Grid.tay_nodes
    hnodes = Grid.height_nodes
    tleft = Grid.tay_left
    #tright = Grid.tay_right
    hup = Grid.height_up
    hdown = Grid.height_down
    dheight = Grid.dheight
    dtay = Grid.dtay
    
    def f(h, Ray, t0, h0, dt, dh):
        J = (R + h) / (np.sqrt(h**2 + 2.*R*h + R**2 * (np.sin(np.deg2rad(Ray.betta)))**2))         
        return J * (h - h0) * (Ray.tay_from_height(h) - t0) / (dt * dh) * (1/np.cos(np.deg2rad(Ray.azimuth)))

    A = np.zeros((len(Rays), tnodes * hnodes))
    #print(np.shape((A)))
    i = 0
    for Ray in Rays:
        dh = 1.
        #betta = Ray.betta
        tay_rec = Ray.tay_rec
        #height_sat = Ray.height_sat            
        t_in = tay_rec
        h_in = hdown
        it_prev = trunc((t_in - tleft) / dtay)
        ih_prev = trunc((h_in - hdown) / dheight)
        for h in np.arange(dh, hup + dh, dh):
            t = Ray.tay_from_height(h)
            it = trunc((t - tleft) / dtay)
            ih = trunc((h - hdown) / dheight)
            if (it != it_prev) or (ih != ih_prev):
                h_out = h

                int1 = quad(f, h_in, h_out, args=(Ray, tleft + (it_prev + 1) * dtay, 
                            hdown + (ih_prev + 1) * dheight, dtay, dheight), epsabs=1e-3)[0] / 10.
                int2 = -quad(f, h_in, h_out, args=(Ray, tleft + it_prev * dtay, 
                            hdown + (ih_prev + 1) * dheight, dtay, dheight), epsabs=1e-3)[0] / 10.
                int3 = -quad(f, h_in, h_out, args=(Ray, tleft + (it_prev + 1) * dtay, 
                            hdown + ih_prev * dheight, dtay, dheight), epsabs=1e-3)[0] / 10.
                int4 = quad(f, h_in, h_out, args=(Ray,  tleft + it_prev * dtay, 
                            hdown + ih_prev * dheight, dtay, dheight), epsabs=1e-3)[0] / 10.

                A[i, it_prev + ih_prev * tnodes] += int1
                A[i, (it_prev + 1) + ih_prev * tnodes] += int2
                A[i, it_prev + (ih_prev + 1) * tnodes] += int3
                A[i, (it_prev + 1) + (ih_prev + 1) * tnodes] += int4
                h_in = h_out
                it_prev = it
                ih_prev = ih

        i = i + 1
        print(i)
    return csr_matrix(A)  # A.tocsr()

def CovMatrixObs(Rays, sigma0=0.05):
    
    """
    Ковариационная матрица наблюдений. Служит для работы фильтра Калмана.
    """
    
    P = lil_matrix((len(Rays),len(Rays)))
    diag = np.zeros(len(Rays))
    for i in range(len(Rays)):
        diag[i] = sigma0**2 / (np.sin(np.deg2rad(Rays[i].betta)))**2
    P.setdiag(diag)
    return P.tocsr()

    
def CovMatrixModel(state, sigma=0.2):
    
    """
    Ковариационная матрица состояния системы. Служит для работы фильтра Калмана.
    """
    
    max_state = max(state)
    diag = sigma**2 * state / max_state
    P = lil_matrix((len(state),len(state)))
    P.setdiag(diag)
    return P.tocsr()
    

def kalman_step(apriory_state, cov_apriory_state, observe, cov_observe, observe_operator):
    
    """
    Первый шаг фильтра Калмана. Корректировка в данной задаче не требуется
    """
    
    S = cov_observe + observe_operator.dot(cov_apriory_state.dot(observe_operator.transpose()))
    K = cov_apriory_state.dot(observe_operator.transpose().dot(inv(S)))
    aposteriory_state = apriory_state + K.dot(observe - observe_operator.dot(apriory_state)) 
    
    return aposteriory_state



def main():

    Model = Background(month=6, hour=12, f10p7=90, lon=20,
                 N_ir_max_arr=[0.5, 1, 1], h0_arr=[300., 300., 300.],
                 tay0_arr=[6700., 6500., 8000], a_arr=[60., 60., 60.],
                 b_arr=[100., 60., 60.], gamma_arr=[0., 45., 0.], v_arr=[1., 1., 1.])

    Model1 = Background(month=9, hour=12, f10p7=120, lon=20)


    MyGrid = TGrid()
    rec_tays = [6200, 7500, 8200] # Координата приёмников
    Rays = CreateRays(rec_tays, MyGrid, data1)

    #RHS = ComputeRHS(Rays, Model)
    

    
    covRHS = CovMatrixObs(Rays)
    
    OperatorH = ObsOperator(MyGrid, Rays)


    apriory_state = np.zeros(MyGrid.tay_nodes * MyGrid.height_nodes)
    
    
    for j in np.arange(MyGrid.height_nodes):
        for i in np.arange(MyGrid.tay_nodes):
            apriory_state[i + j * (MyGrid.tay_nodes)] = Model1.zerostate(MyGrid.tay_left 
                            + i * MyGrid.dtay, MyGrid.height_down + j * MyGrid.dheight)

    apriory_state_plot = apriory_state.reshape((MyGrid.height_nodes, MyGrid.tay_nodes))
    
    cov_apriory_state = CovMatrixModel(apriory_state) #* 1e-15

    Result = kalman_step(apriory_state, cov_apriory_state, data1.TEC, covRHS, OperatorH)
    Result = Result.reshape((MyGrid.height_nodes, MyGrid.tay_nodes))

    ModelN = np.zeros(MyGrid.tay_nodes * MyGrid.height_nodes)
    for j in np.arange(MyGrid.height_nodes):
        for i in np.arange(MyGrid.tay_nodes):
            ModelN[i + j * (MyGrid.tay_nodes)] = Model.model(MyGrid.tay_left
                        + i * MyGrid.dtay, MyGrid.height_down + j * MyGrid.dheight)
    
    
    ModelN = ModelN.reshape((MyGrid.height_nodes, MyGrid.tay_nodes))
    
    
    '''
      ####################################################### PLOTS ###############                               
    '''
###################################################################################

    fig = plt.figure(figsize=(10, 4))
    for Ray in Rays:
        hh = np.arange(0, 1000.)
        tt = Ray.tay_from_height(hh)
        plt.plot(np.rad2deg(tt/R), hh, 'w:', alpha = 0.5)
   
    

    plt.xlim(np.rad2deg(MyGrid.tay_left/R), np.rad2deg(MyGrid.tay_right/R))
    plt.ylim(MyGrid.height_down, MyGrid.height_up)
    plt.title('Model')
    plt.xlabel('latitude, deg')
    plt.ylabel('height, km')

    levels = np.linspace(0., np.max(ModelN), 21)

    im = plt.contourf(ModelN, levels, extent=(np.rad2deg(MyGrid.tay_left/R),
                      np.rad2deg(MyGrid.tay_right/R), MyGrid.height_down, 
                      MyGrid.height_up), extend="both", cmap=cm.jet)
   

    fig.colorbar(im, label='electron density, [$N-unit$]')

    plt.tight_layout()
    plt.show()
##################################################################################
    fig = plt.figure(figsize=(10, 4))
    for Ray in Rays:
        hh = np.arange(0, 1000.)
        tt = Ray.tay_from_height(hh)
        plt.plot(np.rad2deg(tt/R), hh, 'w:', alpha = 0.5)
   
    

    plt.xlim(np.rad2deg(MyGrid.tay_left/R), np.rad2deg(MyGrid.tay_right/R))
    plt.ylim(MyGrid.height_down, MyGrid.height_up)
    plt.title('Kalman')
    plt.xlabel('latitude, deg')
    plt.ylabel('height, km')

    levels = np.linspace(0., np.max(ModelN), 21)

    
    res = plt.contourf(Result, levels, extent=(np.rad2deg(MyGrid.tay_left/R),
                       np.rad2deg(MyGrid.tay_right/R), MyGrid.height_down, 
                       MyGrid.height_up), extend="both", cmap=cm.jet)

    fig.colorbar(res, label='electron density, [$N-unit$]')

    plt.tight_layout()
    plt.show()


 #################################################################################
    
    fig = plt.figure(figsize=(10, 4))
    for Ray in Rays:
        hh = np.arange(0, 1000.)
        tt = Ray.tay_from_height(hh)
        plt.plot(np.rad2deg(tt/R), hh, 'w:', alpha = 0.5)
   
    

    plt.xlim(np.rad2deg(MyGrid.tay_left/R), np.rad2deg(MyGrid.tay_right/R))
    plt.ylim(MyGrid.height_down, MyGrid.height_up)
    plt.title('Apriory State')
    plt.xlabel('latitude, deg')
    plt.ylabel('height, km')

    levels = np.linspace(0., np.max(ModelN), 21)

    
    aprState = plt.contourf(apriory_state_plot, levels, 
                            extent=(np.rad2deg(MyGrid.tay_left/R), 
                            np.rad2deg(MyGrid.tay_right/R), MyGrid.height_down, 
                            MyGrid.height_up), extend="both", cmap=cm.jet)

    fig.colorbar(aprState, label='electron density, [$N-unit$]')

    plt.tight_layout()
    plt.show()

##################################################################################

#    plt.plot(RHS)
#    plt.show()
    
##################################################################################
    
#    fig = plt.figure(figsize=(10, 4))
#    plt.plot(data.TEC)
#    plt.show()

##################################################################################
#    print(np.shape(RHS))

if __name__ == "__main__":
    main()
