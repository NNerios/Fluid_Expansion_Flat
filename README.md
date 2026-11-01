# Fluid_Expansion_Flat
CER-F — Cosmic Evolution Rate


Flatline Expansion Model


Repository: NNerios/Fluid_Expansion_Flat

Framework: Cosmic Evolution Rate (CER)

Branch: CER-F — Flatline Expansion

Originator: Nicholas Nerios

Model status: Independent observational calculation framework



1. Overview


CER-F (Cosmic Evolution Rate — Flatline) is an independent model for calculating the observable expansion rate of the cosmos.


The model is built around a fluid-dynamics interpretation of space: the vacuum is treated conceptually as an extremely dilute fluid substrate through which cosmic expansion is expressed.


Rather than introducing unidentified components to account for observed expansion, CER-F focuses on calculating the expansion history directly from defined parameters and observational measurements.


The framework follows a strict observational principle:


Use identifiable quantities, explicit mathematical relationships, and reproducible calculations. Do not introduce unknown components as established physical inputs.


CER-F does not use dark energy or dark matter as required model components. It does not use general relativity field equations as its computational foundation, nor does it derive its calculations from the ΛCDM framework.


This is not a claim that the equations used by other cosmological frameworks are invalid. CER-F recognizes that a universal mathematical equation can represent an observed relationship while leaving room for independent approaches to interpreting, calculating, and testing that relationship.


The purpose of CER-F is to provide a consistent calculation of observable cosmic expansion and to investigate observational quantities and relationships that may have been overlooked or calculated inconsistently.


2. The Flatline Expansion Equation


The central CER-F equation calculates the expansion rate as a function of redshift:


[
\boxed{
H(z)=H_0\sqrt{\Omega_{vac}+
\Omega_m(1+z)^3}
}
]


The flatline condition is:


[
\boxed{\Omega_m=1-\Omega_{vac}}
]


Substituting this condition gives the operational equation:


[
\boxed{
H(z)=H_0\sqrt{
\Omega_{vac}+
(1-\Omega_{vac})(1+z)^3
}
}
]


This is the strict flatline baseline equation. The exponent is fixed at 3. No additional evolution parameter is fitted in the CER-F core calculation.


2.1 Variables and symbols




Symbol
Definition




(H(z))
Calculated cosmic expansion rate at redshift (z)


(H_0)
Present-day expansion-rate parameter


(z)
Observed cosmological redshift


(\Omega_{vac})
Vacuum contribution parameter


(\Omega_m)
Matter contribution parameter


(1+z)
Redshift scaling factor


(\sqrt{\phantom{x}})
Square-root operation




The parameters (\Omega_{vac}) and (\Omega_m) are dimensionless.


The flatline condition requires:


[
\Omega_{vac}+\Omega_m=1
]


Consequently, only one of these two parameters needs to be independently specified.


2.2 Units


CER-F calculates the expansion rate using the conventional astronomical unit:


[
\boxed{
\mathrm{km,s^{-1},Mpc^{-1}}
}
]


The numerical calculation does not require conversion into seconds as a separate time-domain calculation.


The units are retained as the conventional observational representation of the Hubble expansion rate.


The output (H(z)) is therefore directly comparable with observational Hubble-rate measurements reported in the same units.


3. How to Calculate the CER-F Expansion Rate


The calculation follows a fixed sequence.


Step 1 — Establish the input parameters


Specify:




(H_0): present-day expansion-rate parameter.


(\Omega_{vac}): vacuum contribution parameter.


(z): redshift at which the expansion rate is to be calculated.




Step 2 — Calculate the matter contribution


Apply the flatline condition:


[
\Omega_m=1-\Omega_{vac}
]


This ensures that the two contribution parameters sum to unity.


Step 3 — Calculate the redshift scaling


Calculate:


[
(1+z)^3
]


Step 4 — Calculate the expansion expression


Substitute the parameters into:


[
H(z)=H_0\sqrt{
\Omega_{vac}+
\Omega_m(1+z)^3
}
]


Step 5 — Return the calculated expansion rate


The resulting value is the CER-F predicted expansion rate at the requested redshift.


Step 6 — Compare with observations


Where an observational measurement is available, compare the calculated result with the measured expansion rate.


The difference can be reported as:


[
\Delta H=H_{CER}-H_{obs}
]


The percentage difference can be reported as:


[
\Delta H_{%}=
\frac{H_{CER}-H_{obs}}{H_{obs}}
\times100
]


These comparisons provide a direct way to examine the agreement between the calculated expansion history and observational measurements.


4. Mathematical Implementation


The strict flatline calculation can be implemented as:


import numpy as np

def get_flat_cer_hz(z, H0, Omega_vac):
    """
    CER-F: Strict Flatline Expansion Equation.

    Calculates the observable expansion rate
    as a function of redshift.

    Alpha is fixed at zero.
    No curvature or cyclic terms are included.
    """

    Omega_m = 1.0 - Omega_vac

    H_z = H0 * np.sqrt(
        Omega_vac
        + Omega_m * (1.0 + z)**3
    )

    return H_z



The function accepts a redshift value or an array of redshifts.


It returns the corresponding calculated expansion rate in the same units as (H_0).


The core function deliberately contains no additional adjustable evolution parameter, curvature term, or cyclical expansion term.


5. Observational Comparison


CER-F is intended to be tested against measured expansion data rather than accepted on the basis of mathematical presentation alone.


One important observational resource is the cosmic chronometer dataset, which provides measurements of (H(z)) across a range of redshifts.


The CER-F calculation can be evaluated against these measurements using:




Individual observational comparisons.


Residuals between predicted and measured values.


Measurement uncertainties.


Chi-square statistics.


Reduced chi-square.


Information criteria, where the likelihood and parameter-count assumptions are appropriate.




The dataset, uncertainty treatment, and any covariance matrices must be documented independently.


A diagonal-error calculation and a covariance-aware calculation must remain distinguishable. They are not interchangeable statistical procedures.


The objective is to establish whether the same strict flatline equation can consistently reproduce the available observational expansion measurements.



6. Horizon Calculation — Sound Horizon Module


6.1 Purpose


The horizon calculation is a separate computational module associated with CER-F.


Its purpose is to calculate the sound horizon scale using the expansion history and the propagation speed of sound in the early-universe baryon-photon plasma.


The sound horizon represents the comoving distance that a sound wave can travel before baryons cease to be tightly coupled to photons at the drag epoch.


This provides a calculated distance scale that can subsequently be used in comparisons with baryon acoustic oscillation (BAO) observations.


The horizon module is separate from the core flatline expansion function.


The core CER-F equation calculates (H(z)). The horizon module uses an expansion history to calculate a distance integral.


6.2 Sound Horizon Equation


The sound horizon at the baryon drag epoch is represented by:


[
\boxed{
r_d=
\int_{z_d}^{\infty}
\frac{c_s(z)}{H(z)},dz
}
]


For numerical computation, the upper integration boundary must be replaced by a sufficiently high finite redshift, with convergence checked.


6.3 Variables and symbols




Symbol
Definition




(r_d)
Comoving sound horizon at the drag epoch


(z_d)
Redshift of the baryon drag epoch


(z)
Redshift integration variable


(c_s(z))
Sound speed in the coupled baryon-photon plasma


(H(z))
Expansion rate supplied by the expansion model


(dz)
Redshift integration interval


(\int)
Numerical integration over the specified redshift range




The resulting sound horizon is expressed in megaparsecs (Mpc) when the units of the sound speed, expansion rate, and distance integration are handled consistently.


6.4 Sound Speed


The sound speed of the baryon-photon plasma is represented by:


[
\boxed{
c_s(z)=
\frac{c}{\sqrt{3(1+R(z))}}
}
]


where:


[
R(z)=\frac{3\rho_b(z)}{4\rho_\gamma(z)}
]


Additional variables




Symbol
Definition




(c)
Speed of light


(R(z))
Baryon-to-photon momentum-density ratio


(\rho_b(z))
Baryon density at redshift (z)


(\rho_\gamma(z))
Photon density at redshift (z)




The sound speed depends on the relative baryon and photon densities.


These quantities must be specified or calculated consistently for the horizon integration.


6.5 Horizon Calculation Procedure


The horizon module follows this sequence:


Step 1 — Supply the expansion history


Use the CER-F expansion function to calculate:


[
H(z)
]


over the redshift range required by the integration.


Step 2 — Establish the physical inputs


Supply the baryon and photon density parameters required to calculate the baryon-to-photon ratio.


The current working implementation has used:


[
\Omega_b=0.0493
]


[
\Omega_\gamma=5.4\times10^{-5}
]


These are working input values, not universal constants. Their provenance and suitability must be documented for each calculation.


Step 3 — Calculate the baryon-to-photon ratio


Calculate (R(z)) from the redshift-dependent baryon and photon densities.


Step 4 — Calculate the sound speed


Use:


[
c_s(z)=
\frac{c}{\sqrt{3(1+R(z))}}
]


Step 5 — Determine the drag epoch


Determine (z_d), the redshift at which baryons effectively cease to be tightly coupled to photons.


A physically consistent calculation requires a defined drag-epoch prescription. If a fixed drag redshift is used, it must be identified as an input or approximation rather than presented as an independently calculated result.


Step 6 — Perform the horizon integration


Calculate:


[
r_d=
\int_{z_d}^{z_{\max}}
\frac{c_s(z)}{H(z)},dz
]


Increase (z_{\max}) and check numerical convergence so that the finite integration limit does not materially affect the result.


Step 7 — Return the sound horizon


The module returns the calculated comoving sound horizon in Mpc.


Step 8 — Use the result in BAO comparisons


The calculated (r_d) can be supplied to the separate BAO calculation module.


The horizon calculation does not alter the CER-F core expansion equation.


6.6 Relationship Between the Expansion and Horizon Modules


The two modules have separate responsibilities:




Module
Responsibility




CER-F core
Calculates the expansion rate (H(z))


Horizon module
Calculates the sound horizon (r_d)


BAO module
Uses the distance scale to compare calculated quantities with BAO observations




This separation prevents the sound horizon calculation from becoming an additional adjustable term in the flatline expansion equation.


It also allows the expansion calculation and horizon calculation to be examined independently.



7. Model Boundaries


CER-F is deliberately restricted to the flatline expansion branch.


The following are outside the scope of this repository's core model:




Cyclical expansion and bounce equations.


Curvature extensions.


Variable evolution exponents.


Æ / Unified Reality theory.


Unverified additional fields or parameters.


Dark energy and dark matter as required model components.


General relativity field equations as the computational foundation.


ΛCDM as the source framework for the CER-F calculation.




The model does not claim that every unresolved cosmological observation has already been explained. Its purpose is to calculate, compare, and investigate observable expansion using a clearly defined mathematical framework.


The horizon and BAO calculations are supporting modules, not modifications to the strict flatline equation.



8. Scientific Position


CER-F is based on the premise that observable cosmic expansion should be calculated as directly and consistently as possible.


The fluid interpretation of the vacuum provides the conceptual foundation of the framework. The mathematical model provides the means of calculating its observable expansion history.


The framework does not reject the mathematical validity of equations used by other cosmological models. It recognizes that different physical interpretations and computational approaches may describe the same observed relationships.


CER-F therefore concentrates on independent calculation, uniform treatment of observations, and explicit identification of the quantities being calculated.


Its results must remain open to comparison, reproduction, and external scientific review.


The central objective is straightforward: calculate the observable expansion of the cosmos through a consistent, explicitly defined model, without treating unidentified components as established inputs.



9. Repository Scope


This repository is dedicated to:


CER-F — Cosmic Evolution Rate, Flatline Expansion.


The primary calculation is:


[
H(z)=H_0\sqrt{
\Omega_{vac}+
(1-\Omega_{vac})(1+z)^3
}
]


The horizon module extends the observational calculation by independently evaluating the sound horizon scale.


The strict flatline equation remains the foundation of the model.


CER-F: an independent fluid-based framework for calculating observable cosmic expansion.

