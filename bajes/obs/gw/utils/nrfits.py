from __future__ import division, unicode_literals, absolute_import
import numpy as np

def calc_isco_radius(m , a):
    """
        Calculate the ISCO radius of a Kerr BH
        using eqns. 2.5 and 2.8 from Ori and Thorne,
        Phys Rev D 62, 24022 (2000)

        Parameters
        ----------
        m : mass [solar masses]
        a : Kerr parameter

        Returns
        -------
        ISCO radius
        """

    a = np.minimum(np.array(a),1.) # Only consider a <=1, to avoid numerical problems

    # Ref. Eq. (2.5) of Ori, Thorne Phys Rev D 62 124022 (2000)
    z1 = 1.+(1.-a**2.)**(1./3)*((1.+a)**(1./3) + (1.-a)**(1./3))
    z2 = np.sqrt(3.*a**2 + z1**2)
    a_sign = np.sign(a)
    msun_rad = 1476.6250614046494
    return m*msun_rad*(3+z2 - np.sqrt((3.-z1)*(3.+z1+2.*z2))*a_sign)

def calc_isco_frequency(m , a):
    """
        Calculate the ISCO frequency of a Kerr BH
        using Kepler's law from ISCO radius

        Parameters
        ----------
        m : mass [solar masses]
        a : Kerr parameter

        Returns
        -------
        ISCO frequency
        """
    r_isco = calc_isco_radius(m , a)
    msun_rad = 1476.6250614046494
    c_light  = 299792458.0
    return np.sqrt(msun_rad*m*c_light*c_light/(r_isco**3))/np.pi

def calc_isco_frequency_for_binary(m1 , m2 , a1 , a2):
    """
        Calculate the ISCO frequency of a compact binary
        using parameters of the progenitors black holes.

        Parameters
        ----------
        m1, m2 : mass components [solar masses]
        a1, a2 : spin components

        Returns
        -------
        ISCO frequency
        """
    mf = bbh_final_mass_non_precessing(m1,m2,a1,a2)
    af = bbh_final_spin_non_precessing(m1,m2,a1,a2)
    f_isco = calc_isco_frequency(mf,af)
    return f_isco

def bbh_UIBfits_setup(m1, m2, chi1, chi2):
    """common setup function for UIB final-state and luminosity fit functions
    """

    # Vectorize the function if arrays are provided as input
    m1   = np.vectorize(float)(np.array(m1))
    m2   = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    chi2 = np.vectorize(float)(np.array(chi2))

    if np.any(m1<0):
        raise ValueError("m1 must not be negative")
    if np.any(m2<0):
        raise ValueError("m2 must not be negative")

    if np.any(abs(chi1)>1):
        raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2)>1):
        raise ValueError("chi2 has to be in [-1, 1]")

    # binary masses
    m    = m1+m2
    if np.any(m<=0):
        raise ValueError("m1+m2 must be positive")
    msq  = m*m
    m1sq = m1*m1
    m2sq = m2*m2

    # symmetric mass ratio
    eta  = m1*m2/msq
    if np.any(eta>0.25):
        Warning("Truncating eta from above to 0.25. This should only be necessary in some rounding corner cases, but better check your m1 and m2 inputs...")
        eta = np.minimum(eta,0.25)
    if np.any(eta<0.0):
        Warning("Truncating negative eta to 0.0. This should only be necessary in some rounding corner cases, but better check your m1 and m2 inputs...")
        eta = np.maximum(eta,0.0)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta2*eta2

    # spin variables (in m = 1 units)
    S1    = chi1*m1sq/msq # spin angular momentum 1
    S2    = chi2*m2sq/msq # spin angular momentum 2
    Stot  = S1+S2         # total spin
    Shat  = (chi1*m1sq+chi2*m2sq)/(m1sq+m2sq) # effective spin, = msq*Stot/(m1sq+m2sq)
    Shat2 = Shat*Shat
    Shat3 = Shat2*Shat
    Shat4 = Shat2*Shat2

    # asymmetric spin combination (spin difference), where the paper assumes m1>m2
    # to make our implementation symmetric under simultaneous exchange of m1<->m2 and chi1<->chi2,
    # we flip the sign here when m2>m1
    chidiff  = chi1 - chi2
    if np.any(m2>m1):
        chidiff = np.sign(m1-m2)*chidiff
    chidiff2 = chidiff*chidiff

    # typical squareroots and functions of eta
    sqrt2 = 2.**0.5
    sqrt3 = 3.**0.5
    sqrt1m4eta = (1. - 4.*eta)**0.5

    return m, eta, eta2, eta3, eta4, Stot, Shat, Shat2, Shat3, Shat4, chidiff, chidiff2, sqrt2, sqrt3, sqrt1m4eta

def bbh_final_mass_non_precessing(m1, m2, chi1, chi2, version="v2"):
    """
        Calculate the final mass with the aligned-spin NR fit
        by Xisco Jimenez Forteza, David Keitel, Sascha Husa et al.
        [LIGO-P1600270] [https://arxiv.org/abs/1611.00332]
        versions v1 and v2 use the same ansatz,
        with v2 calibrated to additional SXS and RIT data

        m1, m2: component masses
        chi1, chi2: dimensionless spins of two BHs
        Results are symmetric under simultaneous exchange of m1<->m2 and chi1<->chi2.
        """

    m, eta, eta2, eta3, eta4, Stot, Shat, Shat2, Shat3, Shat4, chidiff, chidiff2, sqrt2, sqrt3, sqrt1m4eta = bbh_UIBfits_setup(m1, m2, chi1, chi2)

    if version == "v1":
        # rational-function Pade coefficients (exact) from Eq. (22) of 1611.00332v1
        b10 = 0.487
        b20 = 0.295
        b30 = 0.17
        b50 = -0.0717
        # fit coefficients from Tables VII-X of 1611.00332v1
        # values at increased numerical precision copied from
        # https://gravity.astro.cf.ac.uk/cgit/cardiff_uib_share/tree/Luminosity_and_Radiated_Energy/UIBfits/LALInference/EradUIB2016_pyform_coeffs.txt
        # git commit 636e5a71462ecc448060926890aa7811948d5a53
        a2 = 0.5635376058169299
        a3 = -0.8661680065959881
        a4 = 3.181941595301782
        b1 = -0.15800074104558132
        b2 = -0.15815904609933157
        b3 = -0.14299315232521553
        b5 = 8.908772171776285
        f20 = 3.8071100104582234
        f30 = 25.99956516423936
        f50 = 1.552929335555098
        f10 = 1.7004558922558886
        f21 = 0.
        d10 = -0.12282040108157262
        d11 = -3.499874245551208
        d20 = 0.014200035799803777
        d30 = -0.01873720734635449
        d31 = -5.1830734185518725
        f11 = 14.39323998088354
        f31 = -232.25752840151296
        f51 = -0.8427987782523847

    elif version == "v2":
        # rational-function Pade coefficients (exact) from Eq. (22) of LIGO-P1600270-v4
        b10 = 0.346
        b20 = 0.211
        b30 = 0.128
        b50 = -0.212
        # fit coefficients from Tables VII-X of LIGO-P1600270-v4
        # values at increased numerical precision copied from
        # https://dcc.ligo.org/DocDB/0128/P1600270/004/FinalStateUIB2016_suppl_Erad_coeffs.txt
        a2 = 0.5609904135313374
        a3 = -0.84667563764404
        a4 = 3.145145224278187
        b1 = -0.2091189048177395
        b2 = -0.19709136361080587
        b3 = -0.1588185739358418
        b5 = 2.9852925538232014
        f20 = 4.271313308472851
        f30 = 31.08987570280556
        f50 = 1.5673498395263061
        f10 = 1.8083565298668276
        f21 = 0.
        d10 = -0.09803730445895877
        d11 = -3.2283713377939134
        d20 = 0.01118530335431078
        d30 = -0.01978238971523653
        d31 = -4.91667749015812
        f11 = 15.738082204419655
        f31 = -243.6299258830685
        f51 = -0.5808669012986468

    else:
        raise ValueError('Unknown version -- should be either "v1" or "v2".')

    # Calculate the radiated-energy fit from Eq. (28) of LIGO-P1600270-v4
    Erad = (((1. + -2.0/3.0*sqrt2)*eta + a2*eta2 + a3*eta3 + a4*eta4)*(1. + b10*b1*Shat*(f10 + f11*eta + (16. - 16.*f10 - 4.*f11)*eta2) + b20*b2*Shat2*(f20 + f21*eta + (16. - 16.*f20 - 4.*f21)*eta2) + b30*b3*Shat3*(f30 + f31*eta + (16. - 16.*f30 - 4.*f31)*eta2)))/(1. + b50*b5*Shat*(f50 + f51*eta + (16. - 16.*f50 - 4.*f51)*eta2)) + d10*sqrt1m4eta*eta2*(1. + d11*eta)*chidiff + d30*Shat*sqrt1m4eta*eta*(1. + d31*eta)*chidiff + d20*eta3*chidiff2

    # Convert to actual final mass
    Mf = m*(1.-Erad)

    return Mf

def bbh_final_spin_non_precessing(m1, m2, chi1, chi2, version="v2"):
    """
        Calculate the final spin with the aligned-spin NR fit
        by Xisco Jimenez Forteza, David Keitel, Sascha Husa et al.
        [LIGO-P1600270] [https://arxiv.org/abs/1611.00332]
        versions v1 and v2 use the same ansatz,
        with v2 calibrated to additional SXS and RIT data

        m1, m2: component masses
        chi1, chi2: dimensionless spins of two BHs
        Results are symmetric under simultaneous exchange of m1<->m2 and chi1<->chi2.
    """

    m, eta, eta2, eta3, eta4, Stot, Shat, Shat2, Shat3, Shat4, chidiff, chidiff2, sqrt2, sqrt3, sqrt1m4eta = bbh_UIBfits_setup(m1, m2, chi1, chi2)

    if version == "v1":
        # rational-function Pade coefficients (exact) from Eqs. (7) and (8) of 1611.00332v1
        a20 = 5.28
        a30 = 1.27
        a50 = 2.89
        b10 = -0.194
        b20 = 0.075
        b30 = 0.00782
        b50 = -0.527
        # fit coefficients from Tables I-IV of 1611.00332v1
        # values at increased numerical precision copied from
        # https://gravity.astro.cf.ac.uk/cgit/cardiff_uib_share/tree/Luminosity_and_Radiated_Energy/UIBfits/LALInference/FinalSpinUIB2016_pyform_coeffs.txt
        # git commit 636e5a71462ecc448060926890aa7811948d5a53
        a2 = 3.772362507208651
        a3 = -9.627812453422376
        a5 = 2.487406038123681
        b1 = 1.0005294518146604
        b2 = 0.8823439288807416
        b3 = 0.7612809461506448
        b5 = 0.9139185906568779
        f21 = 8.887933111404559
        f31 = 23.927104476660883
        f50 = 1.8981657997557002
        f11 = 4.411041530972546
        f52 = 0.
        d10 = 0.2762804043166152
        d11 = 11.56198469592321
        d20 = -0.05975750218477118
        d30 = 2.7296903488918436
        d31 = -3.388285154747212
        f12 = 0.3642180211450878
        f22 = -40.35359764942015
        f32 = -178.7813942566548
        f51 = -5.556957394513334

    elif version == "v2":
        # rational-function Pade coefficients (exact) from Eqs. (7) and (8) of LIGO-P1600270-v4
        a20 = 5.24
        a30 = 1.3
        a50 = 2.88
        b10 = -0.194
        b20 = 0.0851
        b30 = 0.00954
        b50 = -0.579
        # fit coefficients from Tables I-IV of LIGO-P1600270-v4
        # values at increased numerical precision copied from
        # https://dcc.ligo.org/DocDB/0128/P1600270/004/FinalStateUIB2016_suppl_spin_coeffs.txt
        a2 = 3.8326341618708577
        a3 = -9.487364155598392
        a5 = 2.5134875145648374
        b1 = 1.0009563702914628
        b2 = 0.7877509372255369
        b3 = 0.6540138407185817
        b5 = 0.8396665722805308
        f21 = 8.77367320110712
        f31 = 22.830033250479833
        f50 = 1.8804718791591157
        f11 = 4.409160174224525
        f52 = 0.
        d10 = 0.3223660562764661
        d11 = 9.332575956437443
        d20 = -0.059808322561702126
        d30 = 2.3170397514509933
        d31 = -3.2624649875884852
        f12 = 0.5118334706832706
        f22 = -32.060648277652994
        f32 = -153.83722669033995
        f51 = -4.770246856212403

    else:
        raise ValueError('Unknown version -- should be either "v1" or "v2".')

    # Calculate the fit for the Lorb' quantity from Eq. (16) of LIGO-P1600270-v4
    Lorb = (2.*sqrt3*eta + a20*a2*eta2 + a30*a3*eta3)/(1. + a50*a5*eta) + (b10*b1*Shat*(f11*eta + f12*eta2 + (64. - 16.*f11 - 4.*f12)*eta3) + b20*b2*Shat2*(f21*eta + f22*eta2 + (64. - 16.*f21 - 4.*f22)*eta3) + b30*b3*Shat3*(f31*eta + f32*eta2 + (64. - 16.*f31 - 4.*f32)*eta3))/(1. + b50*b5*Shat*(f50 + f51*eta + f52*eta2 + (64. - 64.*f50 - 16.*f51 - 4.*f52)*eta3)) + d10*sqrt1m4eta*eta2*(1. + d11*eta)*chidiff + d30*Shat*sqrt1m4eta*eta3*(1. + d31*eta)*chidiff + d20*eta3*chidiff2

    # Convert to actual final spin
    chif = Lorb + Stot

    return chif

def bns_postmerger_frequency(label, kappa , mass, nu):
    """
        Calculate the merger and postmerger frequencies with the aligned-spin NR fit
        by Matteo Breschi, Sebastiano Bernuzzi, Francesco Zappa et al.
        [https://arxiv.org/abs/1908.11418]
        this versions is calibrated with BAM and THC data

        label : 1,2,3 or 'm'
        kappa : tidal polarizability
        mass  : total mass [solar masses]
        nu    : symmetric mass ratios
    """
    if label == 1:
        Q  = 0.052182
        n1 = 0.002843
        n2 = 0.0
        d1 = 0.012868
        d2 = 0.0
        c  = 5767.6
    elif label == 2:
        Q  = 7.6356
        n1 = 0.066645
        n2 = 0.000040146
        d1 = 10.949
        d2 = 0.040276
        c  = -52.655
    elif label == 3:
        Q  = 4.5722
        n1 = 0.060385
        n2 = 0.00010661
        d1 = 4.1506
        d2 = 0.027552
        c  = 1875.5
    elif label == 'm':
        Q  = 0.033184
        n1 = 0.0013067
        n2 = 0.0
        d1 = 0.0050064
        d2 = 0.0
        c  = 3199.75613
    else:
        raise ValueError("Unknown label for bns_postmerger_frequency.\n Please use 1, 2, 3 or 'm' (merger) respectively for f_{2-0} , f_2 , f_{2+0} and f_merg")
        exit()
    MTsun = 4.925491025543576e-06

    # f_2-0 fit is not realible for kappa > 500
    x = kappa + c * (1 - 4*nu)
    if kappa > 500 and label == 1:
        x0  = 500 + c * (1 - 4*nu)
        q0  = Q*( (1 + n1*x0 + n2*x0*x0) / (1 + d1*x0 + d2*x0*x0) )
        m0  = Q*( (n1 - d1 + 2*(n2-d2)*x0 + (d1*n2 - d2*n1)*x0*x0) / ((1 + d1*x0 + d2*x0*x0)*(1 + d1*x0 + d2*x0*x0)))
        f = q0 + m0 * (x - x0)
    else:
        f = Q * (1 + n1*x + n2*x*x) / (1 + d1*x + d2*x*x)

    return f/(MTsun*mass)

def bns_postmerger_amplitude(label , kappa, mass, nu):
    if label == 0:
        a = 0.032454
        b = -0.000068029
        c = -6735.8
        x = kappa + c * (1 - 4*nu)
        amp = a + b * x
    elif label == 1:
        a = 0.17657
        b =  -0.000037794
        c = 58542
        x = kappa + c * (1 - 4*nu)
        amp = a + b * x
    elif label == 2:
        a = 0.11601
        b = -0.00017376
        c = -623.09
        x = kappa + c * (1 - 4*nu)
        amp = a + b * x
    elif label == 3:
        a = 0.15894
        b = -0.00017317
        c = 4486.2
        x = kappa + c * (1 - 4*nu)
        amp = a + b * x
    elif label == 'm':
        Q  = 0.3491
        n1 = 0.019272
        n2 = -0.0000043729
        d1 = 0.028266
        d2 = 0.0000093643
        c  = 5215
        x = kappa + c * (1 - 4*nu)
        amp = Q * (1 + n1*x + n2*x*x) / (1 + d1*x + d2*x*x)
    else:
        raise ValueError("Unknown label for bns_postmerger_amplitude. Use 0, 1, 2, 3 or 'm' (merger)")
        exit()

    if amp < 0 :
        amp = 0

    MLsun = 1476.6250614046496
    return amp * MLsun * mass


def bns_postmerger_time(label, kappa, mass, nu):
    if label == 0:
        a = 36.614
        b = 0.093684
        c = -562.36
    elif label == 1:
        a = 83.778
        b = 0.16026
        c = -4951.6
    elif label == 2:
        a = 123.01
        b = 0.30531
        c = -6100.2
    elif label == 3:
        a = 161.54
        b = 0.45635
        c = -6471.9
    elif label == 'e':
        a = 1751.7
        b = 0.81406
        c = 33005
    else:
        raise ValueError("Unknown label for bns_postmerger_time. Use 0, 1, 2, 3 or 'e' (end)")
        exit()

    MTsun = 4.925491025543576e-06
    x = kappa + c * (1 - 4*nu)
    t = (a + b*x)
    return t * mass * MTsun
