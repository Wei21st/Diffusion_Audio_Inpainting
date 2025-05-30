import torch
import numpy as np
import math

import utils.training_utils as utils


class EDM():
    """
        Definition of most of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022)
    """

    def __init__(self, args):
        """
        Args:
            args (dictionary): hydra arguments
            sigma_data (float): 
        """
        self.args=args
        self.schedule_type = getattr(args.diff_params, "schedule_type", "power")
        self.sigma_min = args.diff_params.sigma_min
        self.sigma_max =args.diff_params.sigma_max
        self.P_mean=args.diff_params.P_mean
        self.P_std=args.diff_params.P_std
        self.ro=args.diff_params.ro
        self.ro_train=args.diff_params.ro_train
        self.sigma_data=args.diff_params.sigma_data #depends on the training data!! precalculated variance of the dataset
        #parameters stochastic sampling
        self.Schurn=args.diff_params.Schurn
        self.Stmin=args.diff_params.Stmin
        self.Stmax=args.diff_params.Stmax
        self.Snoise=args.diff_params.Snoise

        self.sigmoid_params = getattr(args.diff_params, "sigmoid", {"start": -3, "end": 3, "tau": 1.0})
        self.cosine_params = getattr(args.diff_params, "cosine", {"start": 0, "end": 1, "tau": 1.0})

        #perceptual filter
        if self.args.diff_params.aweighting.use_aweighting:
            self.AW=utils.FIRFilter(filter_type="aw", fs=args.exp.sample_rate, ntaps=self.args.diff_params.aweighting.ntaps)

    def _sigmoid_schedule(self, t):
        start, end, tau = self.sigmoid_params["start"], self.sigmoid_params["end"], self.sigmoid_params["tau"]
        v_start = 1 / (1 + np.exp(-start / tau))
        v_end = 1 / (1 + np.exp(-end / tau))
        output = 1 / (1 + torch.exp(-(t * (end - start) + start) / tau))
        output = (v_end - output) / (v_end - v_start)
        return output

    def _cosine_schedule(self, t):
        start, end, tau = self.cosine_params["start"], self.cosine_params["end"], self.cosine_params["tau"]
        v_start = math.cos(start * math.pi / 2) ** (2 * tau)
        v_end = math.cos(end * math.pi / 2) ** (2 * tau)
        output = torch.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
        output = (v_end - output) / (v_end - v_start)
        return output
       

    def get_gamma(self, t): 
        """
        Get the parameter gamma that defines the stochasticity of the sampler
        Args
            t (Tensor): shape: (N_steps, ) Tensor of timesteps, from which we will compute gamma
        """
        N=t.shape[0]
        gamma=torch.zeros(t.shape).to(t.device)
        
        #If desired, only apply stochasticity between a certain range of noises Stmin is 0 by default and Stmax is a huge number by default. (Unless these parameters are specified, this does nothing)
        indexes=torch.logical_and(t>self.Stmin , t<self.Stmax)
         
        #We use Schurn=5 as the default in our experiments
        gamma[indexes]=gamma[indexes]+torch.min(torch.Tensor([self.Schurn/N, 2**(1/2) -1]))
        
        return gamma

    def create_schedule(self,nb_steps):
        """
        Define the schedule of timesteps
        Args:
           nb_steps (int): Number of discretized steps
        """
        i=torch.arange(0,nb_steps+1)
        if self.schedule_type == "sigmoid":
            i = i / nb_steps
            gamma = self._sigmoid_schedule(i)
            t = self.sigma_min + (self.sigma_max - self.sigma_min) * gamma
        elif self.schedule_type == "cosine":
            i = i / nb_steps
            gamma = self._cosine_schedule(i)
            t = self.sigma_min + (self.sigma_max - self.sigma_min) * gamma
        else:  # Default to power schedule
            t=(self.sigma_max**(1/self.ro) +i/(nb_steps-1) *(self.sigma_min**(1/self.ro) - self.sigma_max**(1/self.ro)))**self.ro
        t[-1]=0
        return t


    def sample_ptrain(self,N):
        """
        For training, getting t as a normal distribution, folowing Karras et al. 
        I'm not using this
        Args:
            N (int): batch size
        """
        lnsigma=np.random.randn(N)*self.P_std +self.P_mean
        return np.clip(np.exp(lnsigma),self.sigma_min, self.sigma_max) #not sure if clipping here is necessary, but makes sense to me
    
    def sample_ptrain_safe(self,N):
        """
        For training, getting  t according to the same criteria as sampling
        Args:
            N (int): batch size
        """
        a = torch.rand(N)
        if self.schedule_type == "sigmoid":
            # Use the same sigmoid γ(t) and scale it to [sigma_min, sigma_max]
            start = self.sigmoid_params["start"]
            end = self.sigmoid_params["end"]
            tau = self.sigmoid_params["tau"]

            v_start = 1 / (1 + math.exp(-start / tau))
            v_end = 1 / (1 + math.exp(-end / tau))
            gamma = 1 / (1 + torch.exp(-((a * (end - start) + start) / tau)))
            gamma = (v_end - gamma) / (v_end - v_start)
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * gamma

        elif self.schedule_type == "cosine":
            # Cosine γ(t), scaled to [sigma_min, sigma_max]
            start = self.cosine_params["start"]
            end = self.cosine_params["end"]
            tau = self.cosine_params["tau"]

            v_start = math.cos(start * math.pi / 2) ** (2 * tau)
            v_end = math.cos(end * math.pi / 2) ** (2 * tau)
            gamma = torch.cos((a * (end - start) + start) * math.pi / 2) ** (2 * tau)
            gamma = (v_end - gamma) / (v_end - v_start)
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * gamma
        else:  # Default to power-law (ro_train)
            sigma = (self.sigma_max**(1 / self.ro_train) +
                    a * (self.sigma_min**(1 / self.ro_train) - self.sigma_max**(1 / self.ro_train)))**self.ro_train

        return sigma

    def sample_prior(self,shape,sigma):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
            sigma (float): noise level of the noise
        """
        n=torch.randn(shape).to(sigma.device)*sigma
        return n

    def cskip(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        
        """
        return self.sigma_data**2 *(sigma**2+self.sigma_data**2)**-1

    def cout(self,sigma ):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return sigma*self.sigma_data* (self.sigma_data**2+sigma**2)**(-0.5)

    def cin(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (self.sigma_data**2+sigma**2)**(-0.5)

    def cnoise(self,sigma ):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (1/4)*torch.log(sigma)

    def lambda_w(self,sigma):
        return (sigma*self.sigma_data)**(-2) * (self.sigma_data**2+sigma**2)
        
    def denoiser(self, xn , net, sigma):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        if len(sigma.shape)==1:
            sigma=sigma.unsqueeze(-1)
        cskip=self.cskip(sigma)
        cout=self.cout(sigma)
        cin=self.cin(sigma)
        cnoise=self.cnoise(sigma)

        return cskip * xn +cout*net(cin*xn, cnoise)  #this will crash because of broadcasting problems, debug later!

    def prepare_train_preconditioning(self, x, sigma):
        #weight=self.lambda_w(sigma)
        #Is calling the denoiser here a good idea? Maybe it would be better to apply directly the preconditioning as in the paper, even though Karras et al seem to do it this way in their code
        # print(x.shape)
        noise=self.sample_prior(x.shape,sigma)

        cskip=self.cskip(sigma)
        cout=self.cout(sigma)
        cin=self.cin(sigma)
        cnoise=self.cnoise(sigma)

        target=(1/cout)*(x-cskip*(x+noise))

        return cin*(x+noise), target, cnoise


    def loss_fn(self, net, x):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        sigma=self.sample_ptrain_safe(x.shape[0]).unsqueeze(-1).to(x.device)

        input, target, cnoise= self.prepare_train_preconditioning(x, sigma)
        estimate=net(input,cnoise)
        
        error=(estimate-target)

        try:
            #this will only happen if the model is cqt-based, if it crashes it is normal
            if self.args.net.use_cqt_DC_correction:
                error=net.CQTransform.apply_hpf_DC(error) #apply the DC correction to the error as we dont want to propagate the DC component of the error as the network is discarding it. It also applies for the nyquit frequency, but this is less critical.
        except:
            pass 

        #APPLY A-WEIGHTING
        if self.args.diff_params.aweighting.use_aweighting:
            error=self.AW(error)

        #here we have the chance to apply further emphasis to the error, as some kind of perceptual frequency weighting could be
        return error**2, sigma

