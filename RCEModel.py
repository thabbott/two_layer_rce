"""
RCEModel.py

Main class for a dry two-layer RCE model
"""

import numpy as np
import matplotlib.pyplot as plt
import prettyprint as pp
import xarray as xr

class RCEModel:

    def __init__(self):
        """
        Default constructor: runs the model with values typical of RCE over
        subtropical land
        """
        self.sigma = 5.67e-8
        """ Stefan-Boltzmann constant (W m-2) """
        self.ems = 0.5
        """ Emissivity of atmospheric layer (nondim) """
        self.c_s = 2e5
        """ Surface heat capacity (J m-2 K-1) """
        self.c_a = 1004. * 300e2 / 9.80
        """ Atmospheric heat capacity (J m-2 K-1) """
        self.c_k = 0.0015 * 1004. * 1.0 * 1.0
        """ Surface exchange coefficient (W m-2 K-1) """
        self.albedo = 0.3
        """ Surface albedo (nondim) """
        self.S_0 = 1360.
        """ Solar constant (W m-2) """
        self.lat = 20.
        """ Solar latitude (degrees) """
        self.n_day = 288
        """ Time steps per day """
        self.dt = 5. * 60.
        """ Model timestep (s) """
        self.nout_spinup = 12
        """ Timesteps per output during spinup """
        self.nout_equil = 12
        """ Timesteps per output during equilibrated period """
        self.nmax_spinup = 365 * 100 * 288
        """ Maximum number of timesteps to run during spinpup """
        self.n_equil = 100 * 288
        """ Timesteps to run after equilibrating """
        self.tol_equil = 1e-4
        """ Relative tolerance in equlibrated day-to-day energy """
        self.runid = 'model_default'

        # Compute insolation
        self.I_0 = (1. - self.albedo) * \
                self.S_0 * np.cos(np.deg2rad(self.lat))

        # Compute rotation rate
        self.omega_day = 2. * np.pi / (self.dt * self.n_day)

        # Initialize temperatures and time step
        self.T_a = (self.I_0 / self.sigma)**0.25
        self.T_s = (self.I_0 / self.sigma)**0.25
        self.energy = np.inf
        self.N = 0

    def nondim_params(self):
        """
        Calculate nondimensional parameters
        @return $c_a c_s^{-1}$
        @retrun $\epsilon$
        @return $c_k I_0^{-3/4} \sigma^{-1/4}$
        @return $2 \pi I_0^{3/4} \sigma^{1/4} \omega_{day}^{-1} c_s^{-1}$
        """
        return (self.c_a / self.c_s,
                self.ems,
                self.c_k * self.I_0**-0.75 * self.sigma**-0.25,
                2. * np.pi * self.I_0**0.75 * self.sigma**0.25 *
                    self.omega_day**-1. * self.c_s**-1)

    def time(self):
        """
        @return model time
        """
        return self.dt * self.N

    def lw_up_surf(self):
        """
        Compute updward surface longwave flux
        @return $\sigma T_s^4$
        """
        return self.sigma * self.T_s**4.

    def lw_down_surf(self):
        """
        Compute downward longwave at surface
        @return $\epsilon \sigma T_a^4$
        """
        return self.ems * self.sigma * self.T_a**4

    def sw_net_surf(self):
        """
        Compute net shortwave at surface
        @return $(1 - \alpha) S_0 \cos(\phi) \cos(\omega_{day} t)$
        """
        return self.I_0 * max(0, np.cos(self.omega_day * self.time()))

    def surface_flux(self):
        """
        Compute surface flux
        @return $c_k (T_s - T_a) H(T_s - T_a)$
        """
        return self.c_k * max(0., self.T_s - self.T_a)

    def lw_em_atm(self):
        """
        Compute atmospheric longwave emission
        @return $2\epsilon \sigma T_a^4$
        """
        return 2. * self.ems * self.sigma * self.T_a**4.

    def lw_abs_atm(self):
        """
        Compute atmospheric longwave absorption
        @return $\epsilon \sigma T_s^4$
        """
        return self.ems * self.sigma * self.T_s**4.

    def spinup(self):
        """
        Run model spinup
        """
        # Create buffer for computing energy
        energy_buf = 0.

        # Create buffers for computing average tendencies
        lw_up_buf = 0.
        lw_down_buf = 0.
        sw_net_buf = 0.
        sfc_flx_buf = 0.
        sw_net_buf = 0.
        lw_em_buf = 0.
        lw_abs_buf = 0.

        # Create buffers for outputs
        nbuf = int(self.nmax_spinup/self.nout_spinup)
        time_out = np.zeros((nbuf,))
        T_a_out = np.zeros((nbuf,))
        T_s_out = np.zeros((nbuf,))
        lw_up_out = np.zeros((nbuf,))
        lw_down_out = np.zeros((nbuf,))
        sfc_flx_out = np.zeros((nbuf,))
        sw_net_out = np.zeros((nbuf,))
        lw_em_out = np.zeros((nbuf,))
        lw_abs_out = np.zeros((nbuf,))

        # Run model
        ibuf = 0
        iday = 0
        iout = 0
        istep = 0
        while istep < self.nmax_spinup:

            # Set tendencies to 0
            dTsdt = 0.
            dTadt = 0.

            # Compute tendencies, add to buffers
            tend = self.lw_up_surf()
            lw_up_buf -= tend
            dTsdt -= tend

            tend = self.lw_down_surf()
            lw_down_buf += tend
            dTsdt += tend

            tend = self.sw_net_surf()
            sw_net_buf += tend
            dTsdt += tend

            tend = self.surface_flux()
            sfc_flx_buf += tend
            dTsdt -= tend
            dTadt += tend

            tend = self.lw_em_atm()
            lw_em_buf -= tend
            dTadt -= tend

            tend = self.lw_abs_atm()
            lw_abs_buf += tend
            dTadt += tend

            # Update temperatures
            self.T_s += dTsdt * self.dt / self.c_s
            self.T_a += dTadt * self.dt / self.c_a

            # Update energy
            energy_buf += self.c_s * self.T_s + self.c_a * self.T_a

            # Increment counters
            iday += 1
            istep += 1
            self.N += 1
            ibuf += 1

            # Compute and save outputs if time
            if ibuf == self.nout_spinup:
                time_out[iout] = self.time()
                T_a_out[iout] = self.T_a
                T_s_out[iout] = self.T_s
                lw_up_out[iout] = lw_up_buf / ibuf
                lw_up_buf = 0.
                lw_down_out[iout] = lw_down_buf / ibuf
                lw_down_buf = 0.
                sw_net_out[iout] = sw_net_buf / ibuf
                sw_net_buf = 0.
                sfc_flx_out[iout] = sfc_flx_buf / ibuf
                sfc_flx_buf = 0.
                lw_em_out[iout] = lw_em_buf / ibuf
                lw_em_buf = 0.
                lw_abs_out[iout] = lw_abs_buf / ibuf
                lw_abs_buf = 0.
                ibuf = 0
                iout = iout + 1

            # Compute energy if needed
            if iday == self.n_day:
                energy_buf = energy_buf / iday
                delta_energy = np.abs((energy_buf - self.energy) / energy_buf)
                self.energy = energy_buf
                energy_buf = 0.
                iday = 0

                # Stop spinup if equilibrated
                if delta_energy < self.tol_equil:
                    self.save_state(time_out[:iout],
                                    T_a_out[:iout],
                                    T_s_out[:iout],
                                    lw_up_out[:iout],
                                    lw_down_out[:iout],
                                    sw_net_out[:iout],
                                    sfc_flx_out[:iout],
                                    lw_em_out[:iout],
                                    lw_abs_out[:iout])
                    pp.info('Model equilibrated successfully')
                    return
                else:
                    pp.info('Spinup time step %d of %d' % 
                            (self.N, self.nmax_spinup))
                    pp.info('Energy residual %.2e (tolerance %.2e)' %
                            (delta_energy, self.tol_equil))

            # end if iday == self.nday

        # end while istep < self.nmax_spinup

        # If this point is reached, model failed to equilibrate
        self.save_state(time_out[:iout],
                        T_a_out[:iout],
                        T_s_out[:iout],
                        lw_up_out[:iout],
                        lw_down_out[:iout],
                        sw_net_out[:iout],
                        sfc_flx_out[:iout],
                        lw_em_out[:iout],
                        lw_abs_out[:iout], spinup = True)
        pp.error('Model failed to equilibrate.')

    def run(self):
        """
        Run model post-spinup
        """
        # Create buffer for computing energy
        energy_buf = 0.

        # Create buffers for computing average tendencies
        lw_up_buf = 0.
        lw_down_buf = 0.
        sw_net_buf = 0.
        sfc_flx_buf = 0.
        sw_net_buf = 0.
        lw_em_buf = 0.
        lw_abs_buf = 0.

        # Create buffers for outputs
        nbuf = int(self.nmax_spinup/self.nout_spinup)
        time_out = np.zeros((nbuf,))
        T_a_out = np.zeros((nbuf,))
        T_s_out = np.zeros((nbuf,))
        lw_up_out = np.zeros((nbuf,))
        lw_down_out = np.zeros((nbuf,))
        sfc_flx_out = np.zeros((nbuf,))
        sw_net_out = np.zeros((nbuf,))
        lw_em_out = np.zeros((nbuf,))
        lw_abs_out = np.zeros((nbuf,))

        # Run model
        ibuf = 0
        iday = 0
        iout = 0
        istep = 0
        while istep < self.n_equil:

            # Set tendencies to 0
            dTsdt = 0.
            dTadt = 0.

            # Compute tendencies, add to buffers
            tend = self.lw_up_surf()
            lw_up_buf -= tend
            dTsdt -= tend

            tend = self.lw_down_surf()
            lw_down_buf += tend
            dTsdt += tend

            tend = self.sw_net_surf()
            sw_net_buf += tend
            dTsdt += tend

            tend = self.surface_flux()
            sfc_flx_buf += tend
            dTsdt -= tend
            dTadt += tend

            tend = self.lw_em_atm()
            lw_em_buf -= tend
            dTadt -= tend

            tend = self.lw_abs_atm()
            lw_abs_buf += tend
            dTadt += tend

            # Update temperatures
            self.T_s += dTsdt * self.dt / self.c_s
            self.T_a += dTadt * self.dt / self.c_a

            # Update energy
            energy_buf += self.c_s * self.T_s + self.c_a * self.T_a

            # Increment counters
            iday += 1
            istep += 1
            self.N += 1
            ibuf += 1

            # Compute and save outputs if time
            if ibuf == self.nout_equil:
                time_out[iout] = self.time()
                T_a_out[iout] = self.T_a
                T_s_out[iout] = self.T_s
                lw_up_out[iout] = lw_up_buf / ibuf
                lw_up_buf = 0.
                lw_down_out[iout] = lw_down_buf / ibuf
                lw_down_buf = 0.
                sw_net_out[iout] = sw_net_buf / ibuf
                sw_net_buf = 0.
                sfc_flx_out[iout] = sfc_flx_buf / ibuf
                sfc_flx_buf = 0.
                lw_em_out[iout] = lw_em_buf / ibuf
                lw_em_buf = 0.
                lw_abs_out[iout] = lw_abs_buf / ibuf
                lw_abs_buf = 0.
                ibuf = 0
                iout = iout + 1

            # Compute energy if needed
            if iday == self.n_day:
                energy_buf = energy_buf / iday
                delta_energy = np.abs((energy_buf - self.energy) / energy_buf)
                self.energy = energy_buf
                energy_buf = 0.
                iday = 0

                # Warn if model starts to drift
                if delta_energy > self.tol_equil:
                    pp.error('Model is drifting!!')
                    pp.error('Time step %d of %d' % 
                            (istep, self.n_equil))
                    pp.error('Energy residual %.2e (tolerance %.2e)' %
                            (delta_energy, self.tol_equil))
                else:
                    pp.info('Time step %d of %d' % (istep, self.n_equil))
                    pp.info('Energy residual %.2e (tolerance %.2e)' %
                            (delta_energy, self.tol_equil))

            # end if iday == self.nday

        # end while istep < self.n_equil

        # If this point is reached, model failed to equilibrate
        self.save_state(time_out[:iout],
                        T_a_out[:iout],
                        T_s_out[:iout],
                        lw_up_out[:iout],
                        lw_down_out[:iout],
                        sw_net_out[:iout],
                        sfc_flx_out[:iout],
                        lw_em_out[:iout],
                        lw_abs_out[:iout], spinup = False)
        pp.info('Model run finished.')

    def save_state(self, time, T_a, T_s, lw_up, lw_down, sw_net, sfc_flx,
                    lw_em, lw_abs, spinup = True):
        """
        Save model state to NetCDF file
        @param time time coordinate
        @param T_a atmospheric temperature timeseries
        @param T_s surface temperature timeseries
        @param lw_up upward longwave from surface
        @param lw_down downward longwave at surface
        @param sw_net net shortwave at surface
        @param sfc_flx energy flux from surface
        @param lw_em longwave emission from atmosphere
        @param lw_abs longwave absoprtion in atmosphere
        @kwarg spinup whether timeseries are from model spinup
        """
        # Create DataArrays
        data_vars = dict()
        data_vars['T_a'] = xr.DataArray(T_a, coords=[time], dims = ['time'],
                attrs = {'units': 'K', 'desc': 'Air temperature'})
        data_vars['T_s'] = xr.DataArray(T_s, coords=[time], dims = ['time'],
                attrs = {'units': 'K', 'desc': 'Surface temperature'})
        data_vars['lw_up_surf'] = \
                xr.DataArray(lw_up, coords=[time], dims = ['time'],
                        attrs = {'units': 'W m$^{-2}$',
                                 'desc': 'Surface upward longwave'})
        data_vars['lw_down_surf'] = \
                xr.DataArray(lw_down, coords=[time], dims = ['time'],
                        attrs = {'units': 'W m$^{-2}$',
                                 'desc': 'Surface downward longwave'})
        data_vars['sw_net_surf'] = \
                xr.DataArray(sw_net, coords=[time], dims = ['time'],
                        attrs = {'units': 'W m$^{-2}$',
                                 'desc': 'Surface net shortwave'})
        data_vars['sfc_flux'] = \
                xr.DataArray(sfc_flx, coords=[time], dims = ['time'],
                        attrs = {'units': 'W m$^{-2}$',
                                 'desc': 'Surface flux'})
        data_vars['lw_em_atm'] = \
                xr.DataArray(lw_em, coords=[time], dims = ['time'],
                        attrs = {'units': 'W m$^{-2}$',
                                 'desc': 'Longwave atmospheric absorption'})
        data_vars['lw_abs_atm'] = \
                xr.DataArray(lw_abs, coords=[time], dims = ['time'],
                        attrs = {'units': 'W m$^{-2}$',
                                 'desc': 'Longwave atmospheric emission'})
        coords = dict()
        coords['time'] = xr.DataArray(time, coords = [time], dims = ['time'],
                attrs = {'units': 's', 'desc': 'Time'})

        # Create metadata
        meta = self.get_params()

        # Create dataset
        dset = xr.Dataset(data_vars = data_vars,
                          coords = coords,
                          attrs = meta)

        # Save dataset
        fname = self.runid + ('.spinup' if spinup else '') + '.nc'
        dset.to_netcdf(path = fname)

    def get_params(self):
        """
        Add model parameters to a dictionary
        @return a dictionary containing model parameters
        """
        out = dict()
        out['sigma'] = self.sigma
        out['ems'] = self.ems
        out['c_s'] = self.c_s
        out['c_a'] = self.c_a
        out['c_k'] = self.c_k
        out['albedo'] = self.albedo
        out['S_0'] = self.S_0
        out['lat'] = self.lat
        out['n_day'] = self.n_day
        out['dt'] = self.dt
        out['nout_spinup'] = self.nout_spinup
        out['nout_equil'] = self.nout_equil
        out['nmax_spinup'] = self.nmax_spinup
        out['n_equil'] = self.n_equil
        out['tol_equil'] = self.tol_equil
        out['runid'] = self.runid
        out['N'] = self.N
        return out
