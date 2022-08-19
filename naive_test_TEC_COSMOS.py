import numpy as np
from datetime import datetime
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import cm
from math import trunc
import random as rnd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import inv
import pandas as pd

import NeQuick_2

start_time = datetime.now()
# define main constants
R = 6371.  # Earth radius in km
R_sat = 7371.  # Radius of the satellites orbit in km
#Upper boundary of the grid must be equel heigth of the Sattelaite

# define geometry of scaning directions
# height, tau - coordinates in km
# betta - inclination of the scaning ray below satellite trajectory in degrees

directory = ('C:\\Users\\IZMIRAN1\\2003-10-13_COSMOS-2389\\200310131910v.rfm')
data1 = pd.read_table(directory, header = None, sep = '\s+', skiprows = 10)
data1.columns = ['Time', 'Lat', 'Phase', 'Lon', 'Ampl', 'El', 'Az', 'TEC']

def rarefaction_data(data, window = 50):
    m = np.zeros(int(len(data)/window))
    k = 0
    for i in range(len(m)):
        m[i] = np.mean(data[k:k+window])
        k = k+window
        
    return m

data_TEC = rarefaction_data(data1.TEC)
data_Az = rarefaction_data(data1.Az)
data_El = rarefaction_data(data1.El)
data_Lat = rarefaction_data(data1.Lat)





#data_TEC =  data1.TEC.rolling(window = 50).mean()
#data_Az = data1.Az.rolling(window = 50).mean()
#data_El = data1.El.rolling(window = 50).mean()

#print(np.shape(data_El))

#data1 = data1[:10000]
#data_TEC = data[['Lat', 'TEC']]
#data_TEC

def moving_average(interval, windowsize = 50):
    
    window = np.ones(int(windowsize)) / float(windowsize) 
    re = np.convolve(interval, window, 'same')
    
    return re

class TRay:
   
   def __init__(self, tau_rec, height_sat, betta, azimuth):
       self.tau_rec = tau_rec
       self.height_sat = R_sat - R
       self.betta = betta
       self.azimuth = azimuth
       
       
   
  
   def tau_from_height(self, height):
       return self.tau_rec + R * (np.radians(self.betta) - np.arccos(R * 
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
    
    def __init__(self, tau_left=4447.67, tau_right=10008, 
                 height_down=0., height_up=1000., tau_nodes=101, height_nodes=61):
        self.tau_left = tau_left
        self.tau_right = tau_right
        self.height_down = height_down
        self.height_up = height_up
        self.tau_nodes = tau_nodes
        self.height_nodes = height_nodes
        self.dtau = (self.tau_right - self.tau_left) / (self.tau_nodes - 1)
        self.dheight = (self.height_up - self.height_down) / (self.height_nodes - 1)


class Background:
    
    '''
    Задание фона - вычисление модели NeQuick2 и добавление к ней неоднородностей
    
    month - месяц; hour - час UT; f10p7 - F10.7; lon - долгота, град; 
    N_ir_max_arr - добавка к эл. концентрации от неоднородности;
    tau0_arr - координата центра неоднородности; h0_arr - высота центра неоднородности;
    a_arr и b_arr - размеры неоднородности; gamma_arr - угод наклона неоднородности; 
    v_arr - (?)
    '''
    
    def __init__(self, month=6, hour=12, f10p7=90, lon=20,
                 N_ir_max_arr=[0.5, ], h0_arr=[150.,],
                 tau0_arr=[6700.,], a_arr=[60., ],
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
        self.tau0_arr = tau0_arr
        self.a_arr = a_arr
        self.b_arr = b_arr
        self.gamma_arr = gamma_arr
        self.v_arr = v_arr
    
    def zerostate(self, tau, height):
        
        '''
        Вычисление модели NeQuick2. Возвращает N-units
        '''
        
        ne = NeQuick_2.nequick(height, np.rad2deg(tau/R), self.lon, 
                               self.month, self.f10p7, self.hour)

        if ne < 0 or np.isnan(ne):
            ne = 0.
        return ne * 1e-12
    
    def irregularity(self, tau, height, N_ir_max, h0, tau0, a, b, gamma, v):
        
        '''
        Вычисление неоднородностей.
        '''
        
        alpha = ((height - h0) * np.cos(np.radians(gamma)) - 
                 (tau - tau0) * np.sin(np.radians(gamma)))**2. / a**2. + \
                ((height - h0) * np.sin(np.radians(gamma)) + 
                 (tau - tau0) * np.cos(np.radians(gamma)))**2. / b**2.
        if alpha <= 1.:
            result = N_ir_max * (np.cos(np.pi*np.sqrt(alpha) / 2.))**2.
        else: 
            result = 0.
        return result
    
    def model(self, tau, height):
        
        """
        Добавлене неоднородностей к NeQuick2. Возвращает N-units.
        """
        
        result = self.zerostate(tau, height)
        for N_ir_max, h0, tau0, a, b, gamma, v in zip(self.N_ir_max_arr, 
            self.h0_arr, self.tau0_arr, self.a_arr, self.b_arr, self.gamma_arr, self.v_arr):
            if v == 1.:
                result = result + self.irregularity(tau, height, N_ir_max, h0, tau0, a, b, gamma, v)
            if v == 0.:
                result = result - self.zerostate(tau, height) * self.irregularity(
                    tau, height, N_ir_max, h0, tau0, a, b, gamma, v)
        if result <= 0.:
            result = 0.
        return result  # returns eldens in N-unit

####################################################################################


def CreateRays(rec_taus, Grid, data_El, data_Az):
    
    """
    Создание лучей просвечивания приёмников.
    """

    Rays = []

    for rec_tau in rec_taus:
        for b, a in zip(data_El,data_Az) :  # Ширина лучей приёмника, было arange(5, 180, 0.5)
            if(
                TRay(rec_tau, R_sat - R, b,a).tau_from_height(Grid.height_up) < Grid.tau_right
                and TRay(rec_tau, R_sat - R, b,a).tau_from_height(Grid.height_up) > Grid.tau_left
                    ):
                Rays.append(TRay(rec_tau, R_sat - R, b,a))
                
    '''
    условие if не даёт лучам выйти за пределы сформированной сетки
    '''
    print('shape Rays', np.shape(Rays),
          f'shape Az {np.shape(data_Az)}, El {np.shape(data_El)}')
    return Rays


def ComputeRHS(Rays, Model):
    
    """
    Генерация псевдонаблюдений, вычисление ПЭС. Возвращает TECu
    """
    
    RHS = []

    def f1(height, Ray, Model): # Интеграл для вычисления ПЭС с учётом Якобиана преобразования
        return Model.model(Ray.tau_from_height(height), height) \
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
    
    tnodes = Grid.tau_nodes
    hnodes = Grid.height_nodes
    tleft = Grid.tau_left
    #tright = Grid.tau_right
    hup = Grid.height_up
    hdown = Grid.height_down
    dheight = Grid.dheight
    dtau = Grid.dtau
    
    def f(h, Ray, t0, h0, dt, dh):
        J = (R + h) / (np.sqrt(h**2 + 2.*R*h + R**2 * (np.sin(np.deg2rad(Ray.betta)))**2))         
        return J * (h - h0) * (Ray.tau_from_height(h) - t0) / (dt * dh) * (1/np.cos(np.deg2rad(Ray.azimuth)))

    A = np.zeros((len(Rays), tnodes * hnodes))
    #print(np.shape((A)))
    i = 0
    for Ray in Rays:
        dh = 1.
        #betta = Ray.betta
        tau_rec = Ray.tau_rec
        #height_sat = Ray.height_sat            
        t_in = tau_rec
        h_in = hdown
        it_prev = trunc((t_in - tleft) / dtau)
        ih_prev = trunc((h_in - hdown) / dheight)
        for h in np.arange(dh, hup + dh, dh):
            t = Ray.tau_from_height(h)
            it = trunc((t - tleft) / dtau)
            ih = trunc((h - hdown) / dheight)
            if (it != it_prev) or (ih != ih_prev):
                h_out = h

                int1 = quad(f, h_in, h_out, args=(Ray, tleft + (it_prev + 1) * dtau, 
                            hdown + (ih_prev + 1) * dheight, dtau, dheight), epsabs=1e-3)[0] / 10.
                int2 = -quad(f, h_in, h_out, args=(Ray, tleft + it_prev * dtau, 
                            hdown + (ih_prev + 1) * dheight, dtau, dheight), epsabs=1e-3)[0] / 10.
                int3 = -quad(f, h_in, h_out, args=(Ray, tleft + (it_prev + 1) * dtau, 
                            hdown + ih_prev * dheight, dtau, dheight), epsabs=1e-3)[0] / 10.
                int4 = quad(f, h_in, h_out, args=(Ray,  tleft + it_prev * dtau, 
                            hdown + ih_prev * dheight, dtau, dheight), epsabs=1e-3)[0] / 10.

                A[i, it_prev + ih_prev * tnodes] += int1
                A[i, (it_prev + 1) + ih_prev * tnodes] += int2
                A[i, it_prev + (ih_prev + 1) * tnodes] += int3
                A[i, (it_prev + 1) + (ih_prev + 1) * tnodes] += int4
                h_in = h_out
                it_prev = it
                ih_prev = ih

        i = i + 1
        print(i)
    return csr_matrix(A).tocsc()  # A.tocsc()

def CovMatrixObs(Rays, sigma0=0.05):
    
    """
    Ковариационная матрица наблюдений. Служит для работы фильтра Калмана.
    """
    
    P = lil_matrix((len(Rays),len(Rays)))
    diag = np.zeros(len(Rays))
    for i in range(len(Rays)):
        diag[i] = sigma0**2 / (np.sin(np.deg2rad(Rays[i].betta)))**2
    P.setdiag(diag)
    return P.tocsc()

    
def CovMatrixModel(state, sigma=0.2):
    
    """
    Ковариационная матрица состояния системы. Служит для работы фильтра Калмана.
    """
    
    max_state = max(state)
    diag = sigma**2 * state / max_state
    P = lil_matrix((len(state),len(state)))
    P.setdiag(diag)
    return P.tocsc()
    

def kalman_step(apriory_state, cov_apriory_state, observe, cov_observe, observe_operator):
    
    """
    Первый шаг фильтра Калмана. Корректировка в данной задаче не требуется
    """
    
    S = cov_observe + observe_operator.dot(cov_apriory_state.dot(observe_operator.transpose()))

    
    print('shape S: ',np.shape(S), 'shape obs operator, transpose: ', np.shape(observe_operator.transpose()), 
          'shape obs operator: ', np.shape(observe_operator),
          'shape cov_apriory: ', np.shape(cov_apriory_state))
    
    # Ошибка выпадает при вычисление inv(S)
    
    K = cov_apriory_state.dot(observe_operator.transpose().dot(inv(S)))
    
    print('shape K: ', np.shape(K), 'shape obs operator: ', np.shape(observe_operator),
          'shape apriory: ', np.shape(apriory_state), 'shape observe: ', np.shape(observe)) 

    aposteriory_state = apriory_state + K.dot(observe - observe_operator.dot(apriory_state)) 
    
    print('shape aposteriory_state: ', np.shape(aposteriory_state))
    
    return aposteriory_state



def main():

    Model = Background(month=6, hour=12, f10p7=90, lon=20,
                 N_ir_max_arr=[0.5, 1, 1], h0_arr=[300., 300., 300.],
                 tau0_arr=[6700., 6500., 8000], a_arr=[60., 60., 60.],
                 b_arr=[100., 60., 60.], gamma_arr=[0., 45., 0.], v_arr=[1., 1., 1.])

    #Model1 = Background(month=9, hour=12, f10p7=120, lon=20)


    MyGrid = TGrid()
    rec_taus = [8684.] # Координаты приёмников
    Rays = CreateRays(rec_taus, MyGrid, data_El, data_Az)

    #RHS = ComputeRHS(Rays, Model)
    

    
    covRHS = CovMatrixObs(Rays)
    
    OperatorH = ObsOperator(MyGrid, Rays)


    apriory_state = np.zeros(MyGrid.tau_nodes * MyGrid.height_nodes)
    
    
    for j in np.arange(MyGrid.height_nodes):
        for i in np.arange(MyGrid.tau_nodes):
            apriory_state[i + j * (MyGrid.tau_nodes)] = Model.zerostate(MyGrid.tau_left 
                            + i * MyGrid.dtau, MyGrid.height_down + j * MyGrid.dheight)

    apriory_state_plot = apriory_state.reshape((MyGrid.height_nodes, MyGrid.tau_nodes))
    
    cov_apriory_state = CovMatrixModel(apriory_state) #* 1e-15

    Result = kalman_step(apriory_state, cov_apriory_state, data_TEC, covRHS, OperatorH)
    Result = Result.reshape((MyGrid.height_nodes, MyGrid.tau_nodes))

    ModelN = np.zeros(MyGrid.tau_nodes * MyGrid.height_nodes)
    for j in np.arange(MyGrid.height_nodes):
        for i in np.arange(MyGrid.tau_nodes):
            ModelN[i + j * (MyGrid.tau_nodes)] = Model.model(MyGrid.tau_left
                        + i * MyGrid.dtau, MyGrid.height_down + j * MyGrid.dheight)
    
    
    ModelN = ModelN.reshape((MyGrid.height_nodes, MyGrid.tau_nodes))
    
    
    '''
      ################################################### PLOTS #############################################                               
    '''
###################################################################################

    fig = plt.figure(figsize=(10, 4))
    for Ray in Rays:
        hh = np.arange(0, 1000.)
        tt = Ray.tau_from_height(hh)
        plt.plot(np.rad2deg(tt/R), hh, 'w:', alpha = 0.5)
   
    

    plt.xlim(np.rad2deg(MyGrid.tau_left/R), np.rad2deg(MyGrid.tau_right/R))
    plt.ylim(MyGrid.height_down, MyGrid.height_up)
    plt.title('Model')
    plt.xlabel('latitude, deg')
    plt.ylabel('height, km')

    levels = np.linspace(0., np.max(ModelN), 21)

    im = plt.contourf(ModelN, levels, extent=(np.rad2deg(MyGrid.tau_left/R),
                      np.rad2deg(MyGrid.tau_right/R), MyGrid.height_down, 
                      MyGrid.height_up), extend="both", cmap=cm.jet)
   

    fig.colorbar(im, label='electron density, [$N-unit$]')

    plt.tight_layout()
    plt.show()
##################################################################################
    fig = plt.figure(figsize=(10, 4))
    for Ray in Rays:
        hh = np.arange(0, 1000.)
        tt = Ray.tau_from_height(hh)
        plt.plot(np.rad2deg(tt/R), hh, 'w:', alpha = 0.5)
   
    

    plt.xlim(np.rad2deg(MyGrid.tau_left/R), np.rad2deg(MyGrid.tau_right/R))
    plt.ylim(MyGrid.height_down, MyGrid.height_up)
    plt.title('Kalman')
    plt.xlabel('latitude, deg')
    plt.ylabel('height, km')

    levels = np.linspace(0., np.max(ModelN), 21)

    
    res = plt.contourf(Result, levels, extent=(np.rad2deg(MyGrid.tau_left/R),
                       np.rad2deg(MyGrid.tau_right/R), MyGrid.height_down, 
                       MyGrid.height_up), extend="both", cmap=cm.jet)

    fig.colorbar(res, label='electron density, [$N-unit$]')

    plt.tight_layout()
    plt.show()


 #################################################################################
    
    fig = plt.figure(figsize=(10, 4))
    for Ray in Rays:
        hh = np.arange(0, 1000.)
        tt = Ray.tau_from_height(hh)
        plt.plot(np.rad2deg(tt/R), hh, 'w:', alpha = 0.5)
   
    

    plt.xlim(np.rad2deg(MyGrid.tau_left/R), np.rad2deg(MyGrid.tau_right/R))
    plt.ylim(MyGrid.height_down, MyGrid.height_up)
    plt.title('Apriory State')
    plt.xlabel('latitude, deg')
    plt.ylabel('height, km')

    levels = np.linspace(0., np.max(ModelN), 21)

    
    aprState = plt.contourf(apriory_state_plot, levels, 
                            extent=(np.rad2deg(MyGrid.tau_left/R), 
                            np.rad2deg(MyGrid.tau_right/R), MyGrid.height_down, 
                            MyGrid.height_up), extend="both", cmap=cm.jet)

    fig.colorbar(aprState, label='electron density, [$N-unit$]')

    plt.tight_layout()
    plt.show()

##################################################################################

#    plt.plot(RHS)
#    plt.show()
    
##################################################################################
    
    fig = plt.figure(figsize=(10, 4))
    plt.plot(data_Lat,data_TEC)
    plt.show()

##################################################################################
#    print(np.shape(RHS))

if __name__ == "__main__":
    main()

print('execute time: ', datetime.now() - start_time)