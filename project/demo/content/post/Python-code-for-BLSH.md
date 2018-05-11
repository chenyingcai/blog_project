---
title: "Python code for Buy Low and Sell High Strategy"
date: 2018-05-10
categories:
- Stochastic Process
- Python finance
tags:
- Python
- GBM
- CIR
- Optimal stopping
keywords:
- Optimal Stopping
- Chain-rule
- Simulation of SDE
- Python
Mathjax: true
autoThumbnailImage: false
thumbnailImagePosition: "top"
metaAlignment: center
---

This post is to replicate the post [Buy low and Sell high strategy](https://chenyingcai.github.io/2018/04/buy-low-and-sell-high-strategy/) with the **Python tool** [Numpy](http://www.numpy.org/) which is an alterative for the [Matlab](https://fr.mathworks.com/).

<!--more-->

<!-- toc -->

<!-- 实际上, python因其通用性和易理解特性越来受到大家的使用. 特别值得一提的是, 当前最受大家关注的人工智能技术大量地采用python来编译.
 -->

Actually, Python appeals to more and more people for its versatility and easy-to-understand features. It is worth mentioning that the artificial intelligence technology that attracts most attention at present is heavily compiled with Python.

As we have mentioned above, we're going to involve the [Numpy](http://www.numpy.org/) into our project here, so coding as following in your terminal to install the [Numpy](http://www.numpy.org/)

```sh
pip3 install numpy
```

We suggest the above solution for installing [Numpy](http://www.numpy.org/). Or there are alterative choice that we search the corresponding package in [Pypi](https://pypi.org/project/). After installing [Numpy](http://www.numpy.org/), we could use the [Numpy](http://www.numpy.org/) in our project by the following code

```python
import numpy as np
```

Now, [Numpy](http://www.numpy.org/) will help us approaching to almost all the [Matlab]()\'s science computation job.

# GBM ( Geometric Brownian Motion ) Model

## Generation of Sample Trajectories (GBM)

<!-- 我们首先是进行GBM的构建样本数据 -->

Firstly, We simulating the sample data of GBM. We construct a `Python Class` to figure out all the works involving GBM 

{{< codeblock "geometric_brownian_motion" "python" "https://chenyingcai.github.io/code/Python-code-for-BLSH/geometric_brownian_motion.py" "geometric_brownian_motion.py" >}}
import numpy as np
import matplotlib.pyplot as plt

class geometric_brownian_motion:
    def __init__(self,x0,mu,sigma,T,trials,N=100):
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.trials = trials
        self.N = int(T*N)
        self.Delta = 1.0/N
        self.T = T
        self.GBM = False
        
    def exact_solution(self, GBM_Plot=False):
        from math import sqrt
        
        __dW = np.empty((self.trials, self.N + 1),dtype=float)
        __dW[:,0] = 0.0
        
        __t = np.empty((self.trials,self.N+1),dtype = float)
        __t_for_theory = np.linspace(0.0,self.T,self.N + 1)
        self.t = __t_for_theory
        
        for __k in range(self.trials):
            __t[__k]= __t_for_theory
        self.t_matrix = __t
        # __delta determines the "speed" of the Brownian motion.  The random variable
        # of the position at time t, X(t), has a normal distribution whose mean is
        # the position at time t=0 and whose variance is delta**2*t.
        __delta = 1.0
        np.random.seed(100)
        __dB = sqrt(self.Delta) * np.random.standard_normal((self.trials,self.N))
        
        np.cumsum(__dB, axis = 1, out = __dW[:,1:self.N+1])
        
        __GBM = self.x0 * np.exp((self.mu-1./2.*self.sigma**2)*__t+self.sigma*__dW)
               
        __meanGBM = np.mean(__GBM,axis = 0)
        
        self.GBM = __GBM
        
        
        __theory_expected_value = np.exp(self.mu*__t_for_theory)
        
        __averr = np.linalg.norm((__meanGBM - __theory_expected_value),ord=np.inf)
        
        #plot
        if GBM_Plot:
            plt.plot(__t_for_theory,__meanGBM,'b-')
            for __k in range(5):
                plt.plot(__t_for_theory,__GBM[__k],'r--')

            plt.xlabel('t',fontsize = 16)
            plt.ylabel('$X_{t}$',fontsize = 16,rotation = 0,horizontalalignment='right')

            __lg1 = u'mean of {trials} paths'.format(trials = self.trials)
            __lg2 = u'{n} individual path'.format(n = 5)

            plt.legend([__lg1,__lg2])

            plt.title(u'Geometric Brownian motion(e=%.4g)\n %d paths,$\mu$ = %.4g, $\sigma$ = %.4g, $\delta$ = %.4g'
            % ( __averr, self.trials , self.mu, self.sigma, self.Delta)
            )        
            plt.show()
        

    def value_function(self,**arg):
        if isinstance(self.GBM, bool):
            print('You need to run exact_solution first')
            return
        
        # calculate the theorical value of the value function

        __temp_t = np.linspace(0.00, self.T, num=self.N+1)
        __t_matrix = np.ones((self.trials, self.N+1), dtype = np.float)
        for __i in range(self.trials):
            __t_matrix[__i,:] = __temp_t
        # calculate the theorical value of v1 in respect to x0
        __v1_theory = (self.x0-arg['cs'])*(self.x0>=arg['xswitch'])+(arg['psi'](self.x0))*(self.x0<arg['xswitch'])
        __discounted_matrix = np.exp(-arg['discounted'] * __t_matrix)
        __temp_J2_upper = (__discounted_matrix * self.GBM) >= arg['xswitch']
        __temp_J2_lower = (__discounted_matrix * self.GBM) < arg['xswitch']
        __temp_J2_switch = __temp_J2_upper * __discounted_matrix * (self.GBM-arg['cs'])+__temp_J2_lower * __discounted_matrix * arg['psi'](self.GBM)
        __temp_J2 = __discounted_matrix * (-self.GBM-arg['cb']) + __temp_J2_switch
        __temp_J1 = np.exp(-arg['discounted'] * __t_matrix) * (self.GBM-arg['cs'])

        __temp_mean_J2 = np.mean(__temp_J2, axis=0)
        __temp_mean_J1 = np.mean(__temp_J1, axis=0)

        __temp_max_J2 = __temp_J2[0,0]-10
        __tau2 = 0.0
        __temp_max_J1 = __temp_J1[0,0]-10
        __tau1 = 0.0

        for __i in range(self.N+1):
            if __temp_mean_J2[__i] >= __temp_max_J2:
                __temp_max_J2 = __temp_mean_J2[__i]
                __tau2 = __temp_t[__i]
            
            if __temp_mean_J1[__i] >= __temp_max_J1:
                __temp_max_J1 = __temp_mean_J1[__i]
                __tau1 = __temp_t[__i]

        if 'text' in arg.keys() and arg['text']:
            self.__text_html(text="""
            $J_1$最大值: %.4f, 执行点$\\tau_1 $: %.2f<br>
            $J_2$最大值: %.4f, 执行点$\\tau_2 $: %.2f<br>
            $\\textit{v} _{1}^{\\textit{Theory}} $ = %.4f
            """ % (__temp_max_J1, __tau1,__temp_max_J2, __tau2, __v1_theory),
                             HTML=True
                            )
        
        self.GBM = False
        
        if 'output' in arg.keys() and arg['output']:
            return {'j1':__temp_max_J1, 'tau1':__tau1,'j2':__temp_max_J2, 'tau2':__tau2, 'v1':__v1_theory}
    
    def __text_html(self, **arg):
        from IPython.core.display import display
        from IPython.core.display import HTML
        
        if 'HTML' in arg.keys() and arg['HTML']:
            display(HTML(arg['text']))
        else:
            print(arg['text'])
        return
{{< /codeblock >}}

<!-- 在类初始化时, 输入在GBM模型中所需要的参数, 包括初始值, 期望值, 标准平方差, 折价因子等等, 然后函数 exact_solution 根据以下公式生成 GBM 片断(scheme)
 -->
At the time of class initialization, enter the parameters required in the GBM model, including initial values $X_0 $, expectation value $\mu $, standard squared differences$\sigma $, discount factors$\beta $, etc. Then the function *exact_solution* generates a GBM scheme according to the following formula

\begin{equation}
\hat{X}_t=X_0\text{Exp}\left[ \left(\mu \text{  }-\frac{1}{2} \sigma ^2\right) t \delta +\sigma \sum _{j=1}^t \sqrt{\delta }B_i\right]
\label{eq:Total_53}
\end{equation}

where $B_i\sim \mathcal{N}(0,1)$.

Considering b = 0.15, $\sigma $=0.1, $X_0=x=1$ and T = 15, as well as $ \pmb{\textit{trials}} $ and N which denote the number of the sample and the division number regarding to the temporal stepwise. With the following coding and the different observed parameters (i.e. $ \pmb{\textit{trials}} $, N) setting, we simulate different groups of sample's trajectories and the results are given in [Figure 1](#fig:case1). Acoording to the results and the discussion in [Buy low and Sell high strategy](https://chenyingcai.github.io/2018/04/buy-low-and-sell-high-strategy/), we shall set $\pmb{\textit{trials}} $ = 8000 and N = 100 and we could then obtenir the appropriate temporal sequence regarding to GBM and whose error $ \pmb{ \textit{ e}} $ is minimized in this case.

```python
%matplotlib inline

import geometric_brownian_motion

def main():
    # GBM mean.
    __mu = 0.15
    # The Wiener process parameter.
    __sigma = 0.1

    # Total time.
    __T = 15.0

    # Number of trajectories to generate.
    __trials = 1000

    __m = geometric_brownian_motion(1.0,__mu,__sigma,__T,__trials,N=100)
    __m.exact_solution(GBM_Plot=True)
    return
    
if __name__ == "__main__":
    main()

```

<div>
    <a name="fig:case1"></a>
    <tr>
        <td>
            <div class="figure left fig-50">
                <a name="fig:case1_1000_001" class="fancybox" href="https://chenyingcai.github.io/img/python_buylow/case1_1000_01_15.png" data-fancybox-group="group:GBM">
                    <img class="fig-img" src="/img/python_buylow/case1_1000_01_15.png">
                </a>
                <center><b>(a). </b>$ \pmb{e} $ (with $\pmb{\textit{trials}} $=1000, $\delta $=0.01)=0.09863</center>
            </div>
        </td>
        <td>
            <div class="figure right fig-50">
                <a name="fig:case1_1000_0001" class="fancybox" href="https://chenyingcai.github.io/img/python_buylow/case1_1000_001_15.png" data-fancybox-group="group:GBM">
                    <img class="fig-img" src="/img/python_buylow/case1_1000_001_15.png">
                </a>
                <center><b>(b). </b>$ \pmb{e} $ (with $\pmb{\textit{trials}} $=1000, $\delta $=0.001)=0.1853</center>
            </div>
        </td>
    </tr>
    <tr>
        <td>
            <div class="figure left fig-50">
                <a name="fig:case1_4000_001" class="fancybox" href="https://chenyingcai.github.io/img/python_buylow/case1_4000_01_15.png" data-fancybox-group="group:GBM">
                    <img class="fig-img" src="/img/python_buylow/case1_4000_01_15.png">
                </a>
                <center><b>(c). </b>$ \pmb{e} $ (with $\pmb{\textit{trials}} $=4000, $\delta $=0.01)=0.03888</center>
            </div>
        </td>
        <td>
            <div class="figure left fig-50">
                <a name="fig:case1_8000_001" class="fancybox" href="chenyingcai.github.io/img/python_buylow/case1_8000_01_15.png" data-fancybox-group="group:GBM">
                    <img class="fig-img" src="/img/python_buylow/case1_8000_01_15.png">
                </a>
                <center><b>(e). </b>$ \pmb{e} $ (with $\pmb{\textit{trials}} $=8000, $\delta $=0.01)=0.04229</center>
            </div>
        </td>
    </tr>
    <br>    
    <center><b>Figure 1 :</b>The simulation results of $dX _{t} = 0.15 X_t dt + 0.1 X_t dW_t$</center>
</div>

Here, I didn't give the figure with $\pmb{\textit{trials}} $=4000, $\delta $=0.001 and $\pmb{\textit{trials}} $=8000, $\delta $=0.001 because it generate too much sample data such that it takes too many time to compute since my computer has not high performance enough.

## GBM Value Function

In this subsection, we are going to simulate the value function trajectories with different initial value $X_0 $ and fixing other parameters. The following formula indicate the simulation computation of the GBM value funtion.

\begin{equation}
g_1(x) = x - c_s \text{ and } g_2= -x - c_b
\label{eq:Total_13}
\end{equation}

\begin{equation}
\nu _1 = \underset{\tau \in \mathcal{T}}{\sup } \mathbb{E}\left[e^{-\beta \tau
}g_1\left(X _{\tau }^x\right)\right] \text{ and } \nu _2 = \underset{\tau\in
\mathcal{T}}{\sup } \mathbb{E}\left[e^{-\beta \tau }g_2 \left( X _{\tau }^x\right) \right]
\label{eq:Total_14}
\end{equation}

\begin{equation}
\nu (x,i) = \underset{\tau \in \mathcal{T}}{\sup } \mathbb{E}\left[ e^{-\beta  \tau }g_i \left( X _{\tau }^x \right) \mathbf{1} _{\left\\{ \tau <\theta \right\\}} +e^{-\beta  \tau }\nu \left( X _{\tau }^x,j \right) \mathbf{1} _{\left\\{ \tau <\theta \right\\} }+e^{-\beta  \theta }\nu \left( X _{\theta }^x, i\right) \right]
\label{eq:Total_15}
\end{equation}

Then we shall find the difference between the mean of simulation value function and the theorical expectation of value function. The calculation approach of the theorical expectation of value function is discussed in [Section 2](https://chenyingcai.github.io/2018/04/buy-low-and-sell-high-strategy/) in previous post. 

```python
def table_GBM():
    
    # GBM mean.
    __mu = 0.15
    # The Wiener process parameter.
    __sigma = 0.1

    # Total time.
    __T = 15.0

    # Number of realizations to generate.
    __trials = 8000

    for ini in (4.000, 4.1111, 4.2222, 4.3333, 4.4444, 4.5556, 4.6667, 4.7778, 4.8889, 5.000):
        __m = geometric_brownian_motion(ini,__mu,__sigma,__T,__trials,N=100)
        __m.exact_solution(GBM_Plot=False)

        A = 0.9837
        alpha = 4.681
        m=-30.0065
        n=1.0065

        __output = __m.value_function(discounted = 0.151, cs = 0.03, cb = 0.03,xswitch = alpha,
                           psi = lambda x: A*x**n,
                           output=True
                           )
        __temp_text = "\n".join([__temp_text, """
<tr>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
</tr>
        """ % (ini, __output['j1'],__output['tau1'], __output['v1'], abs(__output['j1']-__output['v1']) ,__output['j2'],__output['tau2'], 0.000 ,abs(__output['j2']))
                                ])
    return

table_GBM()
```


<dir>
<a name="tb:table1"></a>
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#999;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#444;background-color:#F7FDFA;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#fff;background-color:#26ADE4;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">$X_0$</th>
    <th class="tg-yw4l">$\nu_1$</th>
    <th class="tg-yw4l">$\tau_1$</th>
    <th class="tg-yw4l">$\nu _{1}^{\textit{Theory}}$</th>
    <th class="tg-yw4l">$e_1$</th>
    <th class="tg-yw4l">$\nu_2$</th>
    <th class="tg-yw4l">$\tau_2$</th>
    <th class="tg-yw4l">$\nu _{2}^{\textit{Theory}}$</th>
    <th class="tg-yw4l">$e_2$</th>
  </tr>
  

<tr>
<td class="tg-yw4l">4.0000</td>
<td class="tg-yw4l">3.9745</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">3.9704</td>
<td class="tg-yw4l">0.0041</td>
<td class="tg-yw4l">0.0105</td>
<td class="tg-yw4l">15.0000</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0105</td>
</tr>
        

<tr>
<td class="tg-yw4l">4.1111</td>
<td class="tg-yw4l">4.0853</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">4.0814</td>
<td class="tg-yw4l">0.0039</td>
<td class="tg-yw4l">0.0102</td>
<td class="tg-yw4l">15.0000</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0102</td>
</tr>
        

<tr>
<td class="tg-yw4l">4.2222</td>
<td class="tg-yw4l">4.1962</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">4.1924</td>
<td class="tg-yw4l">0.0037</td>
<td class="tg-yw4l">0.0099</td>
<td class="tg-yw4l">15.0000</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0099</td>
</tr>
        

<tr>
<td class="tg-yw4l">4.3333</td>
<td class="tg-yw4l">4.3070</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">4.3035</td>
<td class="tg-yw4l">0.0035</td>
<td class="tg-yw4l">0.0096</td>
<td class="tg-yw4l">15.0000</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0096</td>
</tr>
        

<tr>
<td class="tg-yw4l">4.4444</td>
<td class="tg-yw4l">4.4179</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">4.4146</td>
<td class="tg-yw4l">0.0033</td>
<td class="tg-yw4l">0.0092</td>
<td class="tg-yw4l">14.9700</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0092</td>
</tr>
        

<tr>
<td class="tg-yw4l">4.5556</td>
<td class="tg-yw4l">4.5288</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">4.5257</td>
<td class="tg-yw4l">0.0031</td>
<td class="tg-yw4l">0.0089</td>
<td class="tg-yw4l">15.0000</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0089</td>
</tr>
        

<tr>
<td class="tg-yw4l">4.6667</td>
<td class="tg-yw4l">4.6397</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">4.6368</td>
<td class="tg-yw4l">0.0029</td>
<td class="tg-yw4l">0.0086</td>
<td class="tg-yw4l">15.0000</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0086</td>
</tr>
        

<tr>
<td class="tg-yw4l">4.7778</td>
<td class="tg-yw4l">4.7505</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">4.7478</td>
<td class="tg-yw4l">0.0027</td>
<td class="tg-yw4l">0.0082</td>
<td class="tg-yw4l">14.9700</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0082</td>
</tr>
        

<tr>
<td class="tg-yw4l">4.8889</td>
<td class="tg-yw4l">4.8614</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">4.8589</td>
<td class="tg-yw4l">0.0025</td>
<td class="tg-yw4l">0.0079</td>
<td class="tg-yw4l">14.9900</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0079</td>
</tr>
        

<tr>
<td class="tg-yw4l">5.0000</td>
<td class="tg-yw4l">4.9722</td>
<td class="tg-yw4l">3.9100</td>
<td class="tg-yw4l">4.9700</td>
<td class="tg-yw4l">0.0022</td>
<td class="tg-yw4l">0.0076</td>
<td class="tg-yw4l">15.0000</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.0076</td>
</tr>
        
</table>
<center><b>Table 1</b>\: Results of expected value and simulated value with different initial value $X_0$ in GBM model</center>
</dir> 


<!-- 从表格中可以看出, 在不同初始值$X_0 $下计算机产生随机数的实验结果与我们通过理论方程求出的结果误差基本趋于近似. 验证了理论框架下的方程是有效的
 -->
From the [Table 1](#tb:table1) above it can be seen that the experimental results of random numbers generated by the computer with the different initial value $X_0 $ are approximately similar to the results obtained by the theoretical equation. It is verified that the equation under the theoretical framework is valid


# CIR Model

<!-- 我们将使用***cir_model类***进行CIR 模型测试(`./project/buylowsellhigh/cir_model.py`)
 -->
We use `class cir_model` to carry on the further works on the [Cox–Ingersoll–Ross model](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model) ( CIR model )

{{< codeblock "cir_model" "python" "https://chenyingcai.github.io/code/Python-code-for-BLSH/cir_model.py" "cir_model.py" >}}
import numpy as np
import matplotlib.pyplot as plt

class CIR_model:
    def __init__(self,x0,kappa,sigma,T,theta,trials,N=100):
        self.x0 = x0
        self.kappa = kappa
        self.sigma = sigma
        self.theta = theta
        self.trials = trials
        self.N = int(T*N)
        self.Delta = 1.0/N
        self.T = T
        self.CIR = False
        self.t = np.linspace(0.00, self.T, num=self.N+1)
        self.t_matrix = np.ones((self.trials, self.N+1), dtype = np.float)
        for __i in range(trials):
            self.t_matrix[__i,:] = self.t
        
    def EM_chain_rule(self, EM_Plot=False):
        from math import sqrt
        
        np.random.seed(100)

        # __delta determines the "speed" of the Brownian motion.  The random variable
        # of the position at time t, X(t), has a normal distribution whose mean is
        # the position at time t=0 and whose variance is delta**2*t.

        __dB = sqrt(self.Delta) * np.random.standard_normal((self.trials,self.N)) # Brownian increments
        
        self.CIR = {}
        
        self.CIR['em'] = np.zeros((self.trials, self.N + 1), dtype = np.float)
        self.CIR['chain'] = np.zeros((self.trials, self.N + 1), dtype = np.float)
        
        for __i in range(self.trials):
            self.CIR['em'][__i, 0] = self.x0
            self.CIR['chain'][__i,0] = sqrt(self.x0)

            __x_temp_em = self.x0
            __x_temp_chain = sqrt(self.x0)

            for __j in range(self.N):
                __f1 = self.kappa * (self.theta - __x_temp_em)
                __g1 = self.sigma *sqrt(abs(__x_temp_em))
                __x_temp_em = __x_temp_em + __f1 * self.Delta + __g1 * __dB[__i,__j]
                self.CIR['em'][__i, __j+1] = __x_temp_em

                __f2 = (4.0*self.kappa*self.theta - self.sigma**2)/(8.0*__x_temp_chain)-(self.kappa*__x_temp_chain)/2.0
                __g2 = self.sigma/2.0
                __x_temp_chain = __x_temp_chain + __f2 * self.Delta + __g2 * __dB[__i,__j]
                self.CIR['chain'][__i,__j+1] = __x_temp_chain

        __diff1 = np.sqrt(np.abs(self.CIR['em'][:,self.N])) - self.CIR['chain'][:,self.N]
        # calculation of the error between square root of EM and Transforming EM scheme
        __xdiff1 = np.linalg.norm(__diff1, ord=np.inf)
        
        __diff2 = self.CIR['em'][:,self.N] - (self.CIR['chain'][:,self.N]**2)
        # calculation of the error between EM and Transforming back EM scheme
        __xdiff2 = np.linalg.norm(__diff2, ord=np.inf)

        
        #plot
        if EM_Plot:
            # square root of EM and Transforming EM 
            fig,ax = plt.subplots()
            __p1 = plt.plot(self.t ,np.sqrt(self.CIR['em'][0,:]), 'b-')
            __p2 = plt.plot(self.t ,self.CIR['chain'][0,:], 'r--')

            plt.xlabel('t',fontsize = 16)
            plt.ylabel('$V_{t}$',fontsize = 16,rotation = 0,horizontalalignment='right')

            __lg1 = 'Square root of EM'
            __lg2 = 'Transforming EM'

            plt.legend([__lg1,__lg2])

            plt.title(u'Square root of EM and Transforming EM (e=%.4g)\n %d paths,$\\kappa$ = %.4g, $\\theta$ = %.4g, $\\sigma$ = %.4g, $\\delta$ = %.4g'
            % ( __xdiff1, self.trials , self.kappa, self.theta, self.sigma, self.Delta)
            )
            for __i in range(1,4):
                plt.plot(self.t ,np.sqrt(self.CIR['em'][__i,:]), 'b-')
                plt.plot(self.t ,self.CIR['chain'][__i,:], 'r--')

            plt.close(0)

            # EM and Transforming back EM
            fig,ax = plt.subplots()
            __p1 = plt.plot(self.t,self.CIR['em'][0,:], 'b-')
            __p2 = plt.plot(self.t,self.CIR['chain'][0,:]**2, 'r--')

            plt.xlabel('t',fontsize = 16)
            plt.ylabel('$X_{t}$',fontsize = 16,rotation = 0,horizontalalignment='right')

            __lg1 = 'Euler Method scheme'
            __lg2 = 'Transforming back EM'

            plt.legend([__lg1,__lg2])

            plt.title(u'EM and Transforming back EM (e=%.4g)\n %d paths,$\\kappa$ = %.4g, $\\theta$ = %.4g, $\\sigma$ = %.4g, $\\delta$ = %.4g'
            % ( __xdiff2, self.trials , self.kappa, self.theta, self.sigma, self.Delta)
            )
            
            for __i in range(1,4):
                plt.plot(self.t,self.CIR['em'][__i,:], 'b-')
                plt.plot(self.t,self.CIR['chain'][__i,:]**2, 'r--')
            
            plt.show()
        

    def value_function(self,**arg):
        if isinstance(self.CIR, bool):
            print('You need to run exact_solution first')
            return
       
        try:
            __A = arg['A']
            __B = arg['B']
            __x_1 = arg['x1'] # the selling point
            __x_2 = arg['x2'] # the buying point
            __discounted_factor = arg['discounted'] # the discounted factor
            __selling_cost = arg['cs'] # the selling cost
            __buying_cost = arg['cb'] # the buying cost
        except:
            print("check out the input parametres have the A, B, x1, x2, discounted")
        # # set the mpmath precision
        # if 'dps' in arg.keys() and arg['dps']:
        #     mmm.mp.dps = arg['dps']
        #     mmm.mp.pretty = True

        # construct the Apsi and Bphi function
        Apsi = lambda x: __A*self.__hyp1f1(__discounted_factor/self.kappa,
                (2.0*self.kappa*self.theta)/(self.sigma)**2.0,
                2.0*self.kappa*x/self.sigma**2.0)
        Bphi = lambda x: __B*self.__hyperU(__discounted_factor/self.kappa,
            (2.0*self.kappa*self.theta)/(self.sigma)**2.0,
            2.0*self.kappa*x/self.sigma**2.0
            )


        # calculate the theorical value of v1 in respect to x0
        __v1_theory = (Bphi(self.x0)+self.x0-__selling_cost)*(self.x0>=__x_1)+(Apsi(self.x0))*(self.x0<__x_1) # calculate the theorical value of v1 in respect to x0
        __v2_theory = Bphi(self.x0)*(self.x0>__x_2)+(Apsi(self.x0)-self.x0-__buying_cost)*(self.x0<=__x_2) # calculate the theorical value of v2 in respect to x0

        # calculation of the EM scheme of the value function
        __Apsi_Xem = np.zeros((self.trials, self.N+1),dtype = float)
        __Bphi_Xem = np.zeros((self.trials, self.N+1),dtype = float)
        
        for __i in range(self.trials):
            for __j in range(self.N+1):
                __Apsi_Xem[__i,__j] = Apsi(self.CIR['em'][__i,__j])
                __Bphi_Xem[__i,__j] = Bphi(self.CIR['em'][__i,__j])

        __temp_discounted_factor = np.exp(-__discounted_factor * self.t_matrix)

        __temp_J1_upper = __temp_discounted_factor * self.CIR['em'] <= __x_2
        __temp_J1_lower = __temp_discounted_factor * self.CIR['em'] > __x_2
        __temp_J1_switch = __temp_J1_upper * __temp_discounted_factor * (__Apsi_Xem-self.CIR['em']-__buying_cost)+__temp_J1_lower * __temp_discounted_factor * __Bphi_Xem
        __temp_J1 = __temp_discounted_factor * (self.CIR['em']-__selling_cost) + __temp_J1_switch

        
        __temp_J2_upper = __temp_discounted_factor * self.CIR['em'] >= __x_1
        __temp_J2_lower = __temp_discounted_factor * self.CIR['em'] < __x_1
        __temp_J2_switch = __temp_J2_upper * __temp_discounted_factor * (__Bphi_Xem+self.CIR['em']-__selling_cost)+__temp_J2_lower * __temp_discounted_factor * __Apsi_Xem
        __temp_J2 = __temp_discounted_factor * (-self.CIR['em']-__buying_cost) + __temp_J2_switch

        __temp_mean_J2 = np.mean(__temp_J2, axis=0)
        __temp_mean_J1 = np.mean(__temp_J1, axis=0)

        __temp_max_J2 = __temp_J2[0,0]-10
        __tau2 = 0.0
        __temp_max_J1 = __temp_J1[0,0]-10
        __tau1 = 0.0

        for __i in range(self.N+1):
            if __temp_mean_J2[__i] >= __temp_max_J2:
                __temp_max_J2 = __temp_mean_J2[__i]
                __tau2 = self.t[__i]
            
            if __temp_mean_J1[__i] >= __temp_max_J1:
                __temp_max_J1 = __temp_mean_J1[__i]
                __tau1 = self.t[__i]

        if 'text' in arg.keys() and arg['text']:
            self.__text_html(text="""
            $J_1$最大值: %.4f, 执行点$\\tau_1 $: %.2f<br>
            $J_2$最大值: %.4f, 执行点$\\tau_2 $: %.2f<br>
            $\\textit{v} _{1}^{\\textit{Theory}} $ = %.4f
            $\\textit{v} _{1}^{\\textit{Theory}} $ = %.4f
            """ % (__temp_max_J1, __tau1,__temp_max_J2, __tau2, __v1_theory,__v2_theory),
                             HTML=True
                            )
        
        self.CIR = False
        
        if 'output' in arg.keys() and arg['output']:
            return {'j1':__temp_max_J1, 'tau1':__tau1,'j2':__temp_max_J2, 'tau2':__tau2, 'v1':__v1_theory, 'v2':__v2_theory}
    
    def __hyp1f1(self,a,b,z,**arg):
        if 'tol' in arg.keys() and arg['tol']:
            __tol = arg['tol']
        else:
            __tol = 1e-5

        if 'nmin' in arg.keys() and arg['nmin']:
            __nmin = arg['nmin']
        else:
            __nmin = 10

        __term = z*a/b
        __output = 1 + __term
        __n = 1
        __an = a
        __bn = b
        
        while(__n < __nmin) or (abs(__term) > __tol):
            __n = __n + 1
            __an = __an + 1
            __bn = __bn + 1
            __term = z * __term * __an / __bn / __n
            __output = __output + __term

        return __output

    def __hyperU(self,a,b,z,**arg):
        from math import gamma
        return (gamma(1-b)/gamma(a-b+1)) * self.__hyp1f1(a,b,z,**arg)+(gamma(b-1)/gamma(a)) * z**(1-b) * self.__hyp1f1(a-b+1,2-b,z,**arg)


    def __text_html(self, **arg):
        from IPython.core.display import display
        from IPython.core.display import HTML
        
        if 'HTML' in arg.keys() and arg['HTML']:
            display(HTML(arg['text']))
        else:
            print(arg['text'])
        return

{{< /codeblock >}}

## CIR (random walk) Sequence 

Here, I introduce the Euler-Maruyama method and Lamperti transformation to generate the CIR random walk. Then we shall investigate the difference between them. Since the CIR Lamperti transformation is strongly converged to the analytic solution and the Euler-Maruyama scheme is approached to Lamperti transforming back scheme with the small enough temporal stepwise. We could therefore use the Euler-Maruyama scheme as our experimental sample trajectories. Considering $\kappa  = 0.06$, $\vartheta  = 1.1$, $\sigma  = 0.181$, T = 16, $X_0 = 1$, $\beta =0.015$ and the time step $\delta = 0.01$ and $ \pmb{\textit{trials}} $ = 8000 paths

```python

def cir_EM_EM_transforming():
    __kappa = 0.06
    __sigma = 0.181
    __T = 16
    __theta = 1.1
    __trials = 8000
    __N = 100
    
    __model = CIR_model(1.00, __kappa, __sigma, __T, __theta, __trials, __N)
    
    __model.EM_chain_rule(EM_Plot=True)

cir_EM_EM_transforming()

```

We could obtenir the figure result:

<div>
    <a name="fig:cir_em_scheme"></a>    
    <tr>
        <td>
            <div class="figure left fig-50">
                <a name="fig:em_em_chain_rule" class="fancybox" href="https://chenyingcai.github.io/img/python_buylow/em_em_chain_rule.png" data-fancybox-group="group:em_chain_rule">
                <img class="fig-img" src="/img/python_buylow/em_em_chain_rule.png">
                </a>
                <center><b>(a). </b>Comparasion of Square root of EM and Transforming EM </center>
            </div>
        </td>
        <td>
            <div class="figure right fig-50">
                <a name="fig:tranformingback_em" class="fancybox" href="https://chenyingcai.github.io/img/python_buylow/tranformingback_em.png" data-fancybox-group="group:em_chain_rule">
                <img class="fig-img" src="/img/python_buylow/tranformingback_em.png">
                </a>
                <center><b>(b). </b>Comparasion of EM and EM transformed back scheme </center>
            </div>
        </td>
    </tr>
    <center><b>Figure 2 :</b>EM, EM by chain rule and transforming back scheme's simulation results in CIR model</center>
</div>

<!-- 从结果看, 在CIR模型中Euler-Maruyama 方法产生的序列(scheme) 与 通过下述方程(CIR模型方程通过Lamperti transformation后)
 -->
Analyzing the results in [Figure 2](#fig:cir_em_scheme), the square of the scheme generated by the Euler-Maruyama method in the CIR model is nearly consistent with the scheme generated by the following equation (after the Lamperti transformation through the CIR model equation)

$$dV_t = \left( \frac{4\kappa  \vartheta  - \sigma ^2}{8V_t} - \frac{\kappa }{2} V_t \right) dt + \frac{\sigma }{2} dW_t , t\geqslant 0, V_0 = \sqrt{X_0}$$

<!-- 而上述方程是强收敛的, 所以上述方程的EM法产生的序列是强收敛于确切法(exact solution)产生的序列的, 因此我们可以总结为CIR 的EM 法产生的序列与确切法(exact solution) 产生的序列也是近似的, 鉴于CIR模型的确切方程(exact solution) 比较复杂, 我们可以直接使用EM 法产生的序列进行研究(既然两种方法产生的序列(scheme)之间误差不大)
 -->

And the above equation is strongly convergent , so the sequence generated by the EM method of the above equation is strongly convergent to the sequence generated by the exact solution, so we can summarize that the sequence produced by the EM method in CIR model is also approximate to the scheme generated by the exact solution of the model. Considering that the exact solution of the CIR model is relatively complex, we can use the sequence generated by the EM method directly (since the errors betweent are small enough)

## Value Function of CIR

Now, we take the parameters introduced above and set differents initial $X_0 $, then we further give a [Table 2](#tb:table2) in which the results of the experimental value and the theorical value of the CIR model's value function are shown. Furthermore, we shall verify that the CIR framework in Buy Low and Sell high stategy is valid. As we have discussed in [Section 3](https://chenyingcai.github.io/2018/04/buy-low-and-sell-high-strategy/) in previous post, we consider the framework of CIR here as following

\begin{equation}
\omega_1=\left\\{\begin{matrix}
A_1 F_1\left( \frac{\beta }{\kappa }, 
\frac{2 \kappa  \vartheta }{\sigma ^2}, 
\frac{2 \kappa }{\sigma ^2} x \right) &  x < x _1^\* \newline 
B U\left( \frac{\beta }{\kappa }, 
\frac{2 \kappa  \vartheta }{\sigma ^2}, 
\frac{2 \kappa }{\sigma ^2} x\right) + g_1(x) & x\geq x _1^\* 
\end{matrix}\right.
\label{eq:Total_60}
\end{equation}

\begin{equation}
\omega_2=\left\\{\begin{matrix}
A_1 F_1 \left( \frac{\beta }{\kappa }, 
\frac{2 \kappa  \vartheta }{\sigma ^2}, 
\frac{2 \kappa }{\sigma ^2} x\right) + g_2(x) & 
x\leq  x _2^\* \newline
B U\left( \frac{\beta }{\kappa }, \frac{2 \kappa  \vartheta }{\sigma ^2}, \frac{2 \kappa }{\sigma ^2} x\right) & x > x _2^\*
\end{matrix}\right.
\label{eq:Total_61}
\end{equation}

Since $\beta =0.015, \kappa =0.06, \vartheta =1.1, \sigma =0.181$ are given and via smooth-fit principle, we have

\begin{equation}
\begin{matrix}
A _1 F_1 \left( 0.25,4.029,3.662 x _1^\* \right) = B U\left( 0.25,4.029,3.662 
x _1^\* \right) + g_1 \left( x _1^\* \right) \newline
A _1 F _1^{\prime } \left( 0.25,4.029,3.662  x _1^\* \right) = B U ^\prime \left( 0.25,4.029,3.662  x _1^\* \right) + g_1^{\prime } \left( x _1^\* \right) \newline
A _1 F_1 \left( 0.25,4.029,3.662  x _2^\* \right) + g_2 \left( x _2^\* \right) = B U\left( 0.25,4.029,3.662  x _2^\* \right) \newline
A _1 F _1^{\prime } \left( 0.25,4.029,3.662  x _2^\* \right) + g _2^{\prime } \left( x _2^\* \right) = B U ^\prime \left( 0.25,4.029,3.662  x _2^\* \right) 
\end{matrix}
\label{eq:Total_62}
\end{equation}

In order to solve the above system and obtain $A, x_1^\*, B, x_2^\*$, we have to face two problem:

- solve the nonlinear systems above
- solve the computing problem of two [Confluent Hypergeometric Function](https://en.wikipedia.org/wiki/Confluent_hypergeometric_function) $U(a,b,z)$ and $\text{}_1 F_1 \left( a,b,z \right)$

Firstly, we give the following code to construct the computing function of $\text{}_1 F_1 \left( a,b,z \right)$ with the formula:

\begin{equation}
\text{ } _{1} F _{1} \left( a,b,z\right) = \sum _{n=0}^{\infty }{\frac{a^{\left(n \right) }z^n}{b^{\left(n \right) }n!}}
\label{eq:hyp1f1wiki}
\end{equation}

where
$$a ^{\left( 0\right)}=1$$

$$a ^{\left( n\right)}=a\left( a+1\right) \left( a+2\right)\cdots \left( a+n-1\right) $$

```python
def __hyp1f1(self,a,b,z,**arg):
    if 'tol' in arg.keys() and arg['tol']:
        __tol = arg['tol']
    else:
        __tol = 1e-5
    # tol is the accuracy of the 
    if 'nmin' in arg.keys() and arg['nmin']:
        __nmin = arg['nmin']
    else:
        __nmin = 10

    __term = z*a/b
    __output = 1 + __term
    __n = 1
    __an = a
    __bn = b
    
    while(__n < __nmin) or (abs(__term) > __tol):
        __n = __n + 1
        __an = __an + 1
        __bn = __bn + 1
        __term = z * __term * __an / __bn / __n
        __output = __output + __term

    return __output

```

We could calculate the $U(a,b,z)$ with :
$$ U(a,b,z) = \frac{\Gamma (1-b)}{\Gamma (a+1-b)}_1F_1(a,b,z) +
\frac{\Gamma (b-1)}{\Gamma (a)} \(z^{1-b}\) \text{}_1F_1(a+1-b,
2-b,z) $$

and python code is given as:

```python
def __hyperU(self,a,b,z,**arg):
    from math import gamma
    return (gamma(1-b)/gamma(a-b+1)) * self.__hyp1f1(a,b,z,**arg)+(gamma(b-1)/gamma(a)) * z**(1-b) * self.__hyp1f1(a-b+1,2-b,z,**arg)
```

Now, we begin to solve the nonlinear system and coding as following:

{{< codeblock "findcirroot" "python" "https://chenyingcai.github.io/code/Python-code-for-BLSH/findcirroot.py" "findcirroot.py" >}}
# coding=utf-8
import mpmath as mmm

def hyp1f1(a,b,z,**arg):
    if 'tol' in arg.keys() and arg['tol']:
        __tol = arg['tol']
    else:
        __tol = 1e-5
    # tol is the accuracy of the 
    if 'nmin' in arg.keys() and arg['nmin']:
        __nmin = arg['nmin']
    else:
        __nmin = 10

    __term = z*a/b
    __output = 1 + __term
    __n = 1
    __an = a
    __bn = b
    
    while(__n < __nmin) or (abs(__term) > __tol):
        __n = __n + 1
        __an = __an + 1
        __bn = __bn + 1
        __term = z * __term * __an / __bn / __n
        __output = __output + __term

    return __output

def hyperU(a,b,z,**arg):
    from math import gamma
    return (gamma(1-b)/gamma(a-b+1)) * hyp1f1(a,b,z,**arg)+(gamma(b-1)/gamma(a)) * z**(1-b) * hyp1f1(a-b+1,2-b,z,**arg)

def main():
    dicounted_factor = 0.015
    kappa = 0.06
    theta = 1.1
    sigma = 0.181
    cs = 0.015
    cb = 0.025

    a = dicounted_factor/kappa
    b = (2*kappa*theta)/sigma**2
    z = 2*kappa/sigma**2
    
    psi = lambda x: hyp1f1(a,b,z*x)
    phi = lambda x: hyperU(a,b,z*x)
    dpsi = lambda x: (a*z/b)*hyp1f1(a+1,b+1,z*x)
    dphi = lambda x: (-a*z)*hyperU(a+1,b+1,z*x)

    mmm.mp.dps = 9
    mmm.mp.pretty = True

    __result = mmm.findroot([
        lambda __x_1,__x_2,A,B: A*psi(__x_1) - (B * phi(__x_1) + __x_1 - cs),
        lambda __x_1,__x_2,A,B: B*phi(__x_2) - (A*psi(__x_2) - __x_2 - cb),
        lambda __x_1,__x_2,A,B: A*dpsi(__x_1) - (B * dphi(__x_1) + 1),
        lambda __x_1,__x_2,A,B: B*dphi(__x_2) - (A*dpsi(__x_2) - 1)
        ],(1.5, 0.5, 0.001, 0.001),solver='secant', verbose=True)

    print "A = {A}, B = {B}, x_1 = {x_1}, x_2 = {x_2} ".format(A = __result[2], B = __result[3], x_1 = __result[0], x_2 = __result[1])

if __name__ == "__main__":
    main()

{{< /codeblock >}}

By applying the initial value [1.5, 0.5, 0.001, 0.001], we could obtain the results $x_1^\*=1.1122$, $x_2^\*=0.6523$, $A=0.9858$, $B=0.4399$. Similarly, we introduce 10 initial price $x$ ranged from 0.5 to 1.5 and the simulated results are presented in [Table 2](#tb:table2). But here, we just give the result with $ \pmb{\textit{trials}} $ = 100 paths and $\delta  = 0.1$, $T=16$ since my computer's performance is not good enough. You could test the one with higher sample paths and smaller temporal stepsize if your computer is available to do so.

```python
def cir_table():
    
    __kappa = 0.06
    __sigma = 0.181
    __T = 16
    __theta = 1.1
    __trials = 100
    __N = 10
    
    for ini in (0.5000, 0.6111, 0.7222, 0.8333, 0.9444, 1.0556, 1.1667, 1.2778, 1.3889, 1.5000):
        __model = CIR_model(ini, __kappa, __sigma, __T, __theta, __trials, __N)
        __model.EM_chain_rule(EM_Plot=False)

        __output = __model.value_function(discounted = 0.015, 
                                          A = 0.9858, 
                                          B = 0.4399, 
                                          x1 = 1.1122, 
                                          x2 = 0.6523, 
                                          cs = 0.015, 
                                          cb = 0.025, 
                                          output=True
                                         )
        __temp_text = "\n".join([__temp_text, """
<tr>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
<td class=\"tg-yw4l\">%.4f</td>
</tr>
        """ % (ini, __output['j1'],__output['tau1'], __output['v1'], abs(__output['j1']-__output['v1']) ,__output['j2'],__output['tau2'], __output['v2'] ,abs(__output['j2']-__output['v2']))
                                ])

    from IPython.core.display import display
    from IPython.core.display import HTML
    
    display(HTML(__template % __temp_text))
    
    return
    
cir_table()
```

<div>
<a name="tb:table2"></a>
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#999;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#444;background-color:#F7FDFA;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#fff;background-color:#26ADE4;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">$X_0$</th>
    <th class="tg-yw4l">$\nu_1$</th>
    <th class="tg-yw4l">$\tau_1$</th>
    <th class="tg-yw4l">$\nu _{1}^{\textit{Theory}}$</th>
    <th class="tg-yw4l">$e_1$</th>
    <th class="tg-yw4l">$\nu_2$</th>
    <th class="tg-yw4l">$\tau_2$</th>
    <th class="tg-yw4l">$\nu _{2}^{\textit{Theory}}$</th>
    <th class="tg-yw4l">$e_2$</th>
  </tr>
  

<tr>
<td class="tg-yw4l">0.5000</td>
<td class="tg-yw4l">1.1365</td>
<td class="tg-yw4l">5.9000</td>
<td class="tg-yw4l">1.1306</td>
<td class="tg-yw4l">0.0058</td>
<td class="tg-yw4l">0.6056</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.6056</td>
<td class="tg-yw4l">0.0000</td>
</tr>
        

<tr>
<td class="tg-yw4l">0.6111</td>
<td class="tg-yw4l">1.1873</td>
<td class="tg-yw4l">5.8000</td>
<td class="tg-yw4l">1.1750</td>
<td class="tg-yw4l">0.0123</td>
<td class="tg-yw4l">0.5389</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.5389</td>
<td class="tg-yw4l">0.0000</td>
</tr>
        

<tr>
<td class="tg-yw4l">0.7222</td>
<td class="tg-yw4l">1.2433</td>
<td class="tg-yw4l">3.5000</td>
<td class="tg-yw4l">1.2257</td>
<td class="tg-yw4l">0.0177</td>
<td class="tg-yw4l">0.4785</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.4816</td>
<td class="tg-yw4l">0.0031</td>
</tr>
        

<tr>
<td class="tg-yw4l">0.8333</td>
<td class="tg-yw4l">1.3074</td>
<td class="tg-yw4l">3.5000</td>
<td class="tg-yw4l">1.2840</td>
<td class="tg-yw4l">0.0234</td>
<td class="tg-yw4l">0.4257</td>
<td class="tg-yw4l">0.0000</td>
<td class="tg-yw4l">0.4412</td>
<td class="tg-yw4l">0.0155</td>
</tr>
        

<tr>
<td class="tg-yw4l">0.9444</td>
<td class="tg-yw4l">1.3754</td>
<td class="tg-yw4l">3.5000</td>
<td class="tg-yw4l">1.3516</td>
<td class="tg-yw4l">0.0238</td>
<td class="tg-yw4l">0.3831</td>
<td class="tg-yw4l">0.4000</td>
<td class="tg-yw4l">0.4115</td>
<td class="tg-yw4l">0.0284</td>
</tr>
        

<tr>
<td class="tg-yw4l">1.0556</td>
<td class="tg-yw4l">1.4466</td>
<td class="tg-yw4l">3.5000</td>
<td class="tg-yw4l">1.4306</td>
<td class="tg-yw4l">0.0160</td>
<td class="tg-yw4l">0.3548</td>
<td class="tg-yw4l">1.4000</td>
<td class="tg-yw4l">0.3887</td>
<td class="tg-yw4l">0.0338</td>
</tr>
        

<tr>
<td class="tg-yw4l">1.1667</td>
<td class="tg-yw4l">1.5268</td>
<td class="tg-yw4l">0.2000</td>
<td class="tg-yw4l">1.5221</td>
<td class="tg-yw4l">0.0048</td>
<td class="tg-yw4l">0.3331</td>
<td class="tg-yw4l">1.4000</td>
<td class="tg-yw4l">0.3704</td>
<td class="tg-yw4l">0.0373</td>
</tr>
        

<tr>
<td class="tg-yw4l">1.2778</td>
<td class="tg-yw4l">1.6219</td>
<td class="tg-yw4l">0.2000</td>
<td class="tg-yw4l">1.6181</td>
<td class="tg-yw4l">0.0038</td>
<td class="tg-yw4l">0.3161</td>
<td class="tg-yw4l">1.4000</td>
<td class="tg-yw4l">0.3553</td>
<td class="tg-yw4l">0.0392</td>
</tr>
        

<tr>
<td class="tg-yw4l">1.3889</td>
<td class="tg-yw4l">1.7193</td>
<td class="tg-yw4l">0.2000</td>
<td class="tg-yw4l">1.7165</td>
<td class="tg-yw4l">0.0028</td>
<td class="tg-yw4l">0.3053</td>
<td class="tg-yw4l">14.2000</td>
<td class="tg-yw4l">0.3426</td>
<td class="tg-yw4l">0.0374</td>
</tr>
        

<tr>
<td class="tg-yw4l">1.5000</td>
<td class="tg-yw4l">1.8185</td>
<td class="tg-yw4l">0.2000</td>
<td class="tg-yw4l">1.8168</td>
<td class="tg-yw4l">0.0017</td>
<td class="tg-yw4l">0.2979</td>
<td class="tg-yw4l">14.2000</td>
<td class="tg-yw4l">0.3318</td>
<td class="tg-yw4l">0.0339</td>
</tr>
        
</table>
<center><b>Table 2</b>: Results of expected value and simulated value with different initial value $X_0$ in CIR model
</center>
</div>
