from __future__ import division, unicode_literals, absolute_import
import numpy as np

from ..utils import compute_lambda_tilde, compute_delta_lambda, compute_quadrupole_yy

gt = 4.92549094830932e-6
EulerGamma = 0.57721566490153286060

def PhifT6PN(f, M, eta, Lam1, Lam2):
    """ Compute 6PN tidal phase correction
    Eq.(1,4) phttps://arxiv.org/abs/1310.8288]
    --------
    f = frequency series [Hz]
    M = total mass [solar masses]
    eta = symmetric mass ratio [dimensionless]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam1 = primary tidal parameter ell=2 [dimensionless]
    Lam2 = secondary tidal parameter ell=2 [dimensionless]
    """
    v = np.power(np.abs(np.pi*M*f*gt),1./3.)
    v2 = v*v
    v5 = v**5
    v10 = v**10
    v12 = v10*v2
    delta = np.sqrt(1.0 - 4.0*eta)
    m1M = 0.5*(1.0 + delta)
    m2M = 0.5*(1.0 - delta)
    Lam = compute_lambda_tilde(m1M, m2M ,Lam1 , Lam2)
    dLam = compute_delta_lambda(m1M, m2M ,Lam1 , Lam2)
    LO = 3.0/128.0/eta/v5
    return LO*( Lam*v10*(- 39.0/2.0 - 3115.0/64.0*v2) + dLam*6595.0/364.0*v12 )

def PhifT7hPN(f, M, eta, Lama, Lamb):
    """ Compute 7.5PN tidal phase correction
    Appendix B [https://arxiv.org/abs/1203.4352]
    """
    v = np.power(np.abs(np.pi*M*f*gt),1./3.)
    delta = np.sqrt(1.0 - 4.0*eta)
    Xa = 0.5*(1.0 + delta)
    Xb = 0.5*(1.0 - delta)
    Xa2 = Xa*Xa
    Xa3 = Xa2*Xa
    Xa4 = Xa3*Xa
    Xa5 = Xa4*Xa
    Xb2 = Xb*Xb
    Xb3 = Xb2*Xb
    Xb4 = Xb3*Xb
    Xb5 = Xb4*Xb
    v2 = v*v
    v3 = v2*v
    v4 = v3*v
    v5 = v4*v
    #v10 = v**10
    beta221a, beta222a, beta311a, beta331a = 0., 0., 0., 0.
    beta221b, beta222b, beta311b, beta331b = 0., 0., 0., 0.
    kapa = 3.* Lama * Xa4 * Xb
    kapb = 3.* Lamb * Xb4 * Xa
    pNa = -3./(16.*eta)*(12. + Xa/Xb) # -4.*(12. - 11.*Xa)*Xa4 # w\ LO term 3./(128.*eta) factored out
    pNb = -3./(16.*eta)*(12. + Xb/Xa) # -4.*(12. - 11.*Xb)*Xb4 # w\ LO term 3./(128.*eta) factored out
    p1a = 5.*(3179. - 919.*Xa - 2286.*Xa2 + 260.*Xa3)/(672.*(12. - 11.*Xa))
    p1b = 5.*(3179. - 919.*Xb - 2286.*Xb2 + 260.*Xb3)/(672.*(12. - 11.*Xb))
    p2a = -np.pi
    p2b = -np.pi
    p3a = 39927845./508032. - 480043345./9144576.*Xa + 9860575./127008.*Xa2 - 421821905./2286144.*Xa3 + 4359700./35721.*Xa4 - 10578445./285768.*Xa5
    p3a += 5./9.*(1. - 2./3.*Xa)*beta222a + 5./684.*(3. - 13.*Xa + 18.*Xa2 - 8.*Xa3)*beta221a + Xb2*(1.-2.*Xa)*(5./36288.*beta311a + 675./448.*beta331a)
    p3a = p3a/(12. - 11.*Xa)
    p3b = 39927845./508032. - 480043345./9144576.*Xb + 9860575./127008.*Xb2 - 421821905./2286144.*Xb3 + 4359700./35721.*Xb4 - 10578445./285768.*Xb5
    p3b += 5./9.*(1. - 2./3.*Xb)*beta222b + 5./684.*(3. - 13.*Xb + 18.*Xb2 - 8.*Xb3)*beta221b + Xa2*(1.-2.*Xb)*(5./36288.*beta311b + 675./448.*beta331b)
    p3b = p3b/(12. - 11.*Xb)
    p4a = -np.pi*(27719. - 22127.*Xa + 7022.*Xa2 - 10232.*Xa3)/(672.*(12. - 11.*Xa))
    p4b = -np.pi*(27719. - 22127.*Xb + 7022.*Xb2 - 10232.*Xb3)/(672.*(12. - 11.*Xb))
    #LO = 3.0/128.0/eta/v5
    return v5*( kapa*pNa*(1. + p1a*v2 + p2a*v3 + p3a*v4 + p4a*v5) +
                kapb*pNb*(1. + p1b*v2 + p2b*v3 + p3b*v4 + p4b*v5) )


def PhifT7hPNComplete(f, M, eta, Lama, Lamb):
    """ Compute 7.5PN tidal phase correction
    https://arxiv.org/abs/2005.13367
    """
    v = np.power(np.abs(np.pi*M*f*gt),1./3.)
    delta = np.sqrt(1.0 - 4.0*eta)
    Xa = 0.5*(1.0 + delta)
    Xb = 0.5*(1.0 - delta)
    Xa2 = Xa*Xa
    Xa3 = Xa2*Xa
    Xa4 = Xa3*Xa
    Xa5 = Xa4*Xa
    Xb2 = Xb*Xb
    Xb3 = Xb2*Xb
    Xb4 = Xb3*Xb
    Xb5 = Xb4*Xb
    v2 = v*v
    v3 = v2*v
    v4 = v3*v
    v5 = v4*v
    #v10 = v**10
    kapa = 3.* Lama * Xa4 * Xb
    kapb = 3.* Lamb * Xb4 * Xa
    pNa = -3./(16.*eta)*(12. + Xa/Xb) # -4.*(12. - 11.*Xa)*Xa4 # w\ LO term 3./(128.*eta) factored out
    pNb = -3./(16.*eta)*(12. + Xb/Xa) # -4.*(12. - 11.*Xb)*Xb4 # w\ LO term 3./(128.*eta) factored out
    p1a = 5.*(3179. - 919.*Xa - 2286.*Xa2 + 260.*Xa3)/(672.*(12. - 11.*Xa))
    p1b = 5.*(3179. - 919.*Xb - 2286.*Xb2 + 260.*Xb3)/(672.*(12. - 11.*Xb))
    p2a = -np.pi
    p2b = -np.pi
    p3a = -5*(-387973870. + 43246839.*Xa + 174965616.*Xa2 + 158378220.*Xa3 - 20427120.*Xa4 + 4572288.*Xa5)/27433728.
    p3a = p3a/(12. - 11.*Xa)
    p3b = -5*(-387973870. + 43246839.*Xb + 174965616.*Xb2 + 158378220.*Xb3 - 20427120.*Xb4 + 4572288.*Xb5)/27433728.
    p3b = p3b/(12. - 11.*Xb)
    p4a = -np.pi*(27719. - 22415.*Xa + 7598.*Xa2 - 10520.*Xa3)/(672.*(12. - 11.*Xa))
    p4b = -np.pi*(27719. - 22127.*Xb + 7022.*Xb2 - 10232.*Xb3)/(672.*(12. - 11.*Xb))
    #LO = 3.0/128.0/eta/v5
    return v5*( kapa*pNa*(1. + p1a*v2 + p2a*v3 + p3a*v4 + p4a*v5) +
                kapb*pNb*(1. + p1b*v2 + p2b*v3 + p3b*v4 + p4b*v5) )

def PhifQM3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam1=0.0, Lam2=0.0):
    """ Compute post-Newtonian EOS-dependent self-spin term @ 3.5PN for compact binary coalescences
    Eq.(50-52) [https://arxiv.org/abs/1812.07923]
    Uses Love-Q relation of Yunes-Yagi, Eq.(41) [https://arxiv.org/abs/1806.01772]
    --------
    f = frequency series [Hz]
    M = total mass [solar masses]
    eta = symmetric mass ratio [dimensionless]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam1 = primary tidal parameter ell=2 [dimensionless]
    Lam2 = secondary tidal parameter ell=2 [dimensionless]
    """
    #TODO: this implementation needs a check
    if Lam1 == 0. and Lam2 == 0. :  return 0.
    v = np.power(np.abs(np.pi*M*f*gt),1./3.)
    v2 = v*v
    delta = np.sqrt(1.0 - 4.0*eta)
    X1 = 0.5*(1.0 + delta)
    X2 = 0.5*(1.0 - delta)
    at1 = X1*s1z
    at2 = X2*s2z
    at1_2 = at1*at1
    at2_2 = at2*at2
    CQ1 = compute_quadrupole_yy(Lam1) -1. #remove BBH contrib.
    CQ2 = compute_quadrupole_yy(Lam2) -1. #remove BBH contrib.
    a2CQ_p_a2CQ = at1_2*CQ1 + at2_2*CQ2
    a2CQ_m_a2CQ = at1_2*CQ1 - at2_2*CQ2
    PhifQM = -75./(64.*eta) * a2CQ_p_a2CQ /v # LO
    PhifQM += ( (45./16.*eta + 15635./896.) * a2CQ_p_a2CQ + 2215./512 * delta * a2CQ_m_a2CQ )*v/eta # NLO
    PhifQM += -75./(8.*eta) * a2CQ_p_a2CQ * v2 *np.pi # Tail
    return PhifQM

def Phif3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0):
    """ Compute post-Newtonian phase @ 3.5PN for compact binary coalescences
    including spins contributions and tidal effects @ 6PN (if Lam or dLam != 0)
    --------
    f = frequency series [Hz]
    M = binary mass [solar masses]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam = reduced tidal deformability parameter [dimensionless]
    dLam = asymmetric reduced tidal deformation parameter [dimensionless]
    --------
    Adapted from
    https://bitbucket.org/dailiang8/gwbinning/src/master/
    """
    vlso = 1.0/np.sqrt(6.0)
    delta = np.sqrt(1.0 - 4.0*eta)
    v = np.abs(np.pi*M*f*gt)**(1./3.)
    v2 = v*v
    v3 = v2*v
    v4 = v2*v2
    v5 = v4*v
    v6 = v3*v3
    v7 = v3*v4
    v10 = v5*v5
    v12 = v10*v2
    eta2 = eta**2
    eta3 = eta**3

    m1M = 0.5*(1.0 + delta)
    m2M = 0.5*(1.0 - delta)
    chi1L = s1z
    chi2L = s2z
    chi1sq = s1x*s1x + s1y*s1y + s1z*s1z
    chi2sq = s2x*s2x + s2y*s2y + s2z*s2z
    chi1dotchi2 = s1x*s2x + s1y*s2y + s1z*s2z
    SL = m1M*m1M*chi1L + m2M*m2M*chi2L
    dSigmaL = delta*(m2M*chi2L - m1M*chi1L)

    # Phase correction due to spins
    sigma = eta*(721.0/48.0*chi1L*chi2L - 247.0/48.0*chi1dotchi2)
    sigma += 719.0/96.0*(m1M*m1M*chi1L*chi1L + m2M*m2M*chi2L*chi2L)
    sigma -= 233.0/96.0*(m1M*m1M*chi1sq + m2M*m2M*chi2sq)
    phis_15PN = 188.0*SL/3.0 + 25.0*dSigmaL
    ga = (554345.0/1134.0 + 110.0*eta/9.0)*SL + (13915.0/84.0 - 10.0*eta/3.0)*dSigmaL
    pn_ss3 =  (326.75/1.12 + 557.5/1.8*eta)*eta*chi1L*chi2L
    pn_ss3 += ((4703.5/8.4 + 2935.0/6.0*m1M - 120.0*m1M*m1M) + (-4108.25/6.72 - 108.5/1.2*m1M + 125.5/3.6*m1M*m1M))*m1M*m1M*chi1sq
    pn_ss3 += ((4703.5/8.4 + 2935.0/6.0*m2M - 120.0*m2M*m2M) + (-4108.25/6.72 - 108.5/1.2*m2M + 125.5/3.6*m2M*m2M))*m2M*m2M*chi2sq
    phis_3PN = np.pi*(3760.0*SL + 1490.0*dSigmaL)/3.0 + pn_ss3
    phis_35PN = ( -8980424995.0/762048.0 + 6586595.0*eta/756.0 - 305.0*eta2/36.0)*SL - (170978035.0/48384.0 - 2876425.0*eta/672.0 - 4735.0*eta2/144.0)*dSigmaL

    # Point mass
    LO = 3.0/128.0/eta/v5
    pointmass = 1.0
    pointmass += 20.0/9.0*(743.0/336.0 + 11.0/4.0*eta)*v2
    pointmass += (phis_15PN - 16.0*np.pi)*v3
    pointmass += 10.0*(3058673.0/1016064.0 + 5429.0/1008.0*eta + 617.0/144.0*eta2 - sigma)*v4
    pointmass += (38645.0/756.0*np.pi - 65.0/9.0*eta*np.pi - ga)*(1.0 + 3.0*np.log(v/vlso))*v5
    pointmass += (11583231236531.0/4694215680.0 - 640.0/3.0*np.pi**2 - 6848.0/21.0*(EulerGamma + np.log(4.0*v)) +  (-15737765635.0/3048192.0 + 2255.0*np.pi**2/12.0)*eta + 76055.0/1728.0*eta2 - 127825.0/1296.0*eta3 + phis_3PN)*v6
    pointmass += (np.pi*(77096675.0/254016.0 + 378515.0/1512.0*eta - 74045.0/756.0*eta**2) + phis_35PN)*v7

    # Tidal correction to phase at 6PN
    # Eq.(1,4) [https://arxiv.org/abs/1310.8288]
    # Lam is the reduced tidal deformation parameter (\tilde\Lambda)
    # dLam is the asymmetric reduced tidal deformation parameter (\delta\tilde\Lambda)
    tidal = 0.
    if Lam != 0. or dLam != 0. :
        tidal += Lam*v10*(- 39.0/2.0 - 3115.0/64.0*v2) + dLam*6595.0/364.0*v12

    return LO*( pointmass + tidal )


def Phif5hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0):
    """ Compute post-Newtonian phase @ 5.5PN for compact binary coalescences
    including spins contributions and tidal effects @ 6PN (if Lam or dLam != 0)
    [https://arxiv.org/abs/1904.09558]
    --------
    f = frequency series [Hz]
    M = binary mass [solar masses]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam = reduced tidal deformability parameter [dimensionless]
    dLam = asymmetric reduced tidal deformation parameter [dimensionless]
    """

    vlso = 1.0/np.sqrt(6.0)
    delta = np.sqrt(1.0 - 4.0*eta)
    v = (np.pi*M*f*gt)**(1.0/3.0)
    v2 = v*v
    v3 = v2*v
    v4 = v2*v2
    v5 = v4*v
    v6 = v3*v3
    v7 = v3*v4
    v8 = v7*v
    v9 = v8*v
    v10 = v5*v5
    v11 = v10*v
    logv = np.log(v)
    eta2 = eta**2
    eta3 = eta**3
    log2 = 0.69314718055994528623
    log3 = 1.0986122886681097821

    phi_35pn = Phif3hPN(f, M, eta, s1x, s1y, s1z, s2x, s2y, s2z, Lam, dLam)

    c_21_3PN = 0.
    c_22_4PN = 0.
    c_22_5PN = 0.
    a_6_c    = 0.

    coef_8pn = c_21_3PN * (4./8.1 * eta - 16./8.1 * eta2) + c_22_4PN * 160./9. * eta - 36946947827.5/1601901100.8 * eta * eta * eta * eta + 51004148102.5/1310646355.2 * eta3 + (30060067316599.7/57668439628.8 - 39954.5/2721.6 * np.pi * np.pi) * eta * eta + (-567987228950352.7/128152088064. - 532292.8/396.9 * EulerGamma + 930221.5/5443.2 * np.pi * np.pi - 142068.8/44.1 * log2 + 2632.5/4.9 * log3) * eta - 9049./56.7 * np.pi * np.pi - 3681.2/18.9 * EulerGamma + 255071384399888515.3/83042553065472. -2632.5/19.6 * log3 - 101102./396.9 * log2

    coef_log8pn = -3 *(c_21_3PN*(4.0/8.1 * eta - 16./8.1 * eta * eta) + c_22_4PN * 160./9. *eta - 36946947827.5/1601901100.8 * eta * eta * eta * eta + 51004148102.5/1310646355.2 * eta * eta * eta + (30060067316599.7/57668439628.8 - 39954.5/2721.6 * np.pi * np.pi) * eta * eta + (-567987228950352.7/128152088064. - 532292.8/396.9 * EulerGamma + 930221.5/5443.2 * np.pi * np.pi - 142068.8/44.1 * log2 + 2632.5/ 4.9 * log3) * eta - 9049./56.7 * np.pi * np.pi - 3681.2/18.9 * EulerGamma + 255071384399888515.3/83042553065472. -2632.5/19.6 * log3 - 101102./396.9 * log2)

    coef_loglog8pn = 9 * (266146.4/1190.7 * eta + 1840.6/56.7 )

    coef_9pn = np.pi * (1032375.5/19958.4 * eta * eta * eta + 4529333.5/12700.8 * eta * eta + (2255.0/6.0 * np.pi * np.pi - 149291726073.5/13412044.8)* eta -640.0/3.0 * np.pi * np.pi - 1369.6/2.1 * EulerGamma + 10534427947316.3/1877686272.0 - 2739.2/2.1 * log2)

    coef_log9pn = -3 * 1369.6/6.3 * np.pi

    coef_10pn = 1.0/(1.0 - 3.0 * eta) * (a_6_c * (72.0 * eta - 216.0 * eta * eta) + c_21_3PN * (-76.4/2.1 * eta * eta * eta * eta - 59.9/6.3 * eta * eta * eta + 281.5/18.9 * eta * eta - 48.4/18.9 * eta) + c_22_4PN * ( 2564.0/7.0 * eta *eta *eta - 69.8/2.1 * eta * eta - 62.2/2.1 * eta) + c_22_5PN *(48.0*eta*eta -16.0*eta ) + (242506658510205297979.7/85723270481616768.0) * eta * eta * eta * eta * eta * eta - (1272143474037195162.1/67631771583129.6)* eta * eta * eta * eta * eta + (1116081080066315514991.3/27213736660830720.0 -943479.7/1881.6*np.pi * np.pi) * eta * eta * eta * eta + (-85710407655931086054085.1/3428930819264670720.0 -614779314.2/152806.5 * EulerGamma - 46051.9/153.6 * np.pi * np.pi -4311179766.8/152806.5 * log2 + 127939.5/9.8*log3) * eta * eta * eta + (-1873639936380505730110521.7/36575262072156487680.0 - (9923919211.9/458419.5) * EulerGamma + 41579551.7/90316.8 * np.pi * np.pi - 11734037971.3/458419.5*log2 - 5833093.5/548.8*log3)*eta*eta + (56993518125966874478111.3/1083711468804636672.0 + 6378740752.7/916839.0 * EulerGamma - 545142954.7/812851.2 * np.pi * np.pi + 15994339707.7/1833678.0 * log2 + (892417.5/313.6)*log3) * eta + (57822311.5/304819.2) * np.pi * np.pi + (647058264.7/2750517.0)*EulerGamma - 143300652329540712655.9/12630669799587840.0 - 551245.5/2195.2*log3 + 5399283943.1/5501034.0*log2)

    coef_log10pn = 3.0/(1.0 - 3.0 * eta) * (1286378036.2/458419.5 * eta * eta * eta + 1384949312.9/1375258.5 * eta * eta - 2427943164.1/2750517.0 * eta + 647058264.7/8251551.0)

    coef_11pn = np.pi * ( c_21_3PN * (-16.0/2.7 * eta * eta + 4.0/2.7 * eta) + 32.0/9.0 * eta * c_22_4PN + 65762707344.5/14417109907.2 *eta *eta *eta *eta - 108059782847.5/2621292710.4 * eta * eta * eta + 512031495514639.7/62911025049.6 * eta * eta +  (-1064790.5/3628.8 * eta * eta + 4501578.5/14515.2 * eta - 9439.0/56.7) * np.pi * np.pi + (-134666.2/56.7 * EulerGamma - 43038370739839704.7/3460106377728.0 + 2632.5/4.9 * log3 - 2100962.6/396.9 * log2) *eta - 355801.1/793.8 * EulerGamma + 185754140723659441.1/27680851021824.0 - 2632.5/19.6 * log3 - 86254.9/113.4 * log2)

    coef_log11pn = -3* np.pi * (134666.2/170.1 * eta + 355801.1/2381.4)

    return phi_35pn + (3.0/128.0/eta/v5)*((coef_8pn + coef_log8pn*logv + coef_loglog8pn*logv*logv) * v8 +
                                          (coef_9pn + coef_log9pn*logv) * v9 +
                                          (coef_10pn + coef_log10pn*logv) * v10 +
                                          (coef_11pn + coef_log11pn*logv) * v11 )


def Af3hPN(f, M, eta, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam=0.0, dLam=0.0, Deff=1.):
    """ Compute post-Newtonian amplitude @ 3.5PN for compact binary coalescences
    --------
    f = frequency series [Hz]
    M = binary mass [solar masses]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam = reduced tidal deformability parameter [dimensionless] (not used)
    dLam = asymmetric reduced tidal deformation parameter [dimensionless] (not used)
    Deff = luminosity distance
    --------
    Adapted from
    https://bitbucket.org/dailiang8/gwbinning/src/master/
    """

    Mchirp = M*np.power(np.abs(eta),3./5.)
    delta = np.sqrt(1.0 - 4.0*eta)
    v = np.power(np.abs(np.pi*M*f*gt),1./3.)
    v2 = v*v
    v3 = v2*v
    v4 = v2*v2
    v5 = v4*v
    v6 = v3*v3
    v7 = v3*v4
    eta2 = eta**2
    eta3 = eta**3

    # 0PN order
    A0 = np.power(np.abs(Mchirp),5./6.)/np.power(np.abs(f),7./6.)/Deff/np.abs(np.pi)**(2.0/3.0)*np.sqrt(5.0/24.0)

    # Modulus correction due to aligned spins
    chis = 0.5*(s1z + s2z)
    chia = 0.5*(s1z - s2z)
    be = 113.0/12.0*(chis + delta*chia - 76.0/113.0*eta*chis)
    sigma = chia**2*(81.0/16.0 - 20.0*eta) + 81.0/8.0*chia*chis*delta + chis**2*(81.0/16.0 - eta/4.0)
    eps = delta*chia*(502429.0/16128.0 - 907.0/192.0*eta) + chis*(5.0/48.0*eta2 - 73921.0/2016.0*eta + 502429.0/16128.0)

    return A0*(1.0 + v2*(11.0/8.0*eta + 743.0/672.0) + v3*(be/2.0 - 2.0*np.pi) + v4*(1379.0/1152.0*eta2 + 18913.0/16128.0*eta + 7266251.0/8128512.0 - sigma/2.0)
               + v5*(57.0/16.0*np.pi*eta - 4757.0*np.pi/1344.0 + eps)
               + v6*(856.0/105.0*EulerGamma + 67999.0/82944.0*eta3 - 1041557.0/258048.0*eta2 - 451.0/96.0*np.pi**2*eta + 10.0*np.pi**2/3.0
                     + 3526813753.0/27869184.0*eta - 29342493702821.0/500716339200.0 + 856.0/105.0*np.log(4.0*v))
               + v7*(- 1349.0/24192.0*eta2 - 72221.0/24192.0*eta - 5111593.0/2709504.0)*np.pi)


# Combine the modulus and the phase of h(f)
def TaylorF2(phaseorder, f, M, q, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, Lam1=0.0, Lam2=0.0,  Deff=1.0, iota=0.0 ,f_min=20., phiRef=0.0, timeShift=0.0, tidalorder=12, usequadrupolemonopole=0, usenewtides=0):
    """ Compute post-Newtonian frequency-domain waveform @ 3.5PN or 5.5PN for compact binary coalescences,
    including spins contributions and tidal effects @ 6PN or 7.5PN
    Maximum frequency is ISCO (identically zero above).
    --------
    f = frequency series [Hz]
    M = total mass [solar masses]
    eta = symmetric mass ratio [dimensionless]
    s1x = primary spin component along x axis [dimensionless]
    s1y = primary spin component along y axis [dimensionless]
    s1z = primary spin component along z axis [dimensionless]
    s2x = secondary spin component along x axis [dimensionless]
    s2y = secondary spin component along y axis [dimensionless]
    s2z = secondary spin component along z axis [dimensionless]
    Lam1 = primary tidal parameter ell=2 [dimensionless]
    Lam2 = secondary tidal parameter ell=2 [dimensionless]
    Deff = luminosiry distance [Mpc]
    iota = inclination angle [rad]
    Fmin = initial frequency [Hz]
    phiRef = reference phase [rad]
    timeShift = time-shift [sec]
    --------
    hp = array, comples
    plus polarization
    hc = array, comples
    cross polarization
    --------
    Adapted from
    https://bitbucket.org/dailiang8/gwbinning/src/master/
    """
    pre = 3.6686934875530996e-19     # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec
    eta = q / ((1.+q)*(1.+q))

    # Point mass phase
    # Note Lam = 0 = dLam in the following calls
    if phaseorder == 7:
        Phi = Phif3hPN(f, M, eta, s1x,s1y,s1z, s2x,s2y,s2z)
    elif phaseorder == 11:
        Phi = Phif5hPN(f, M, eta, s1x,s1y,s1z, s2x,s2y,s2z)
    else:
        # default: 5.5PN
        Phi = Phif5hPN(f, M, eta, s1x,s1y,s1z, s2x,s2y,s2z)

    # Tidal and QM contributions
    PhiT = 0.
    if Lam1 != 0. or Lam2 != 0. :
        # Tidal terms
        if tidalorder == 12:
            PhiT = PhifT6PN(f, M, eta, Lam1, Lam2)
        elif tidalorder == 15:
            if usenewtides == 1:
                PhiT = PhifT7hPNComplete(f, M, eta, Lam1, Lam2)
            else:
                PhiT = PhifT7hPN(f, M, eta, Lam1, Lam2)
        else:
            # default: 6PN #TODO: set 7.5PN when checked
            PhiT = PhifT6PN(f, M, eta, Lam1, Lam2)
        # Quadrupole-monopole term
        # [https://arxiv.org/abs/gr-qc/9709032]
        PhiQM = 0.
        if usequadrupolemonopole:
            PhiQM = PhifQM3hPN(f, M, eta,
                               s1x,s1y,s1z,
                               s2x,s2y,s2z,
                               Lam1, Lam2)
        PhiT += PhiQM

    # Shift phase
    Phi += PhiT + phiRef + 2*np.pi*f*timeShift

    # Amplitude (no tidal correction implemented)
    A = Af3hPN(f, M, eta,
               s1x=s1x, s1y=s1y, s1z=s1z,
               s2x=s2x, s2y=s2y, s2z=s2z,
               Lam=0.0, dLam=0.0, Deff=Deff)

    # h+, hx
    # Note the convention for the sign in front of the phase
    cosi        = np.cos(iota)
    incl_plus   = (1+cosi*cosi)*0.5
    incl_cross  = cosi
    h   = pre*A*np.exp(-1j*Phi)
    hp  = h * incl_plus
    hc  = h * np.exp(-1j*np.pi/2) * incl_cross
    return hp , hc

def taylorf2_35pn_wrapper(freqs , params):
    """ Wrapper for TF2 3.5PN + (6PN tides) , (if Lam1 or Lam2 != 0)
    """
    phaseorder = 7
    tidalorder = 12
    return TaylorF2(phaseorder, freqs, params['mtot'], params['q'],
                    params['s1x'], params['s1y'], params['s1z'],
                    params['s2x'], params['s2y'], params['s2z'],
                    params['lambda1'], params['lambda2'],
                    params['distance'], params['iota'] ,
                    params['f_min'], params['phi_ref'], 0.,
                    tidalorder=tidalorder)


def taylorf2_55pn_wrapper(freqs , params):
    """ Wrapper for TF2 5.5PN + (6PN tides) , (if Lam1 or Lam2 != 0)
    """
    phaseorder = 11
    tidalorder = 12
    return TaylorF2(phaseorder, freqs, params['mtot'], params['q'],
                    params['s1x'], params['s1y'], params['s1z'],
                    params['s2x'], params['s2y'], params['s2z'],
                    params['lambda1'], params['lambda2'],
                    params['distance'], params['iota'] ,
                    params['f_min'], params['phi_ref'], 0.,
                    tidalorder=tidalorder)


def taylorf2_55pn75pntides_wrapper(freqs , params):
    """ Wrapper for TF2 5.5PN + (7.5PN tides) , (if Lam1 or Lam2 != 0)
    """
    phaseorder = 11
    tidalorder = 15
    return TaylorF2(phaseorder, freqs, params['mtot'], params['q'],
                    params['s1x'], params['s1y'], params['s1z'],
                    params['s2x'], params['s2y'], params['s2z'],
                    params['lambda1'], params['lambda2'],
                    params['distance'], params['iota'] ,
                    params['f_min'], params['phi_ref'], 0.,
                    tidalorder=tidalorder)


def taylorf2_55pn75pnnewtides_wrapper(freqs , params):
    """ Wrapper for TF2 5.5PN + (7.5PN new tides from Henry et al. 2020) , (if Lam1 or Lam2 != 0)
        """
    phaseorder = 11
    tidalorder = 15
    return TaylorF2(phaseorder, freqs, params['mtot'], params['q'],
                    params['s1x'], params['s1y'], params['s1z'],
                    params['s2x'], params['s2y'], params['s2z'],
                    params['lambda1'], params['lambda2'],
                    params['distance'], params['iota'] ,
                    params['f_min'], params['phi_ref'], 0.,
                    tidalorder=tidalorder, usenewtides=1)

def taylorf2_55pn35pnqm75pntides_wrapper(freqs , params):
    """ Wrapper for TF2 5.5PN + (3.5 PN QM) + (7.5PN tides) , (if Lam1 or Lam2 != 0)
    """
    phaseorder = 11
    tidalorder = 15
    return TaylorF2(phaseorder, freqs, params['mtot'], params['q'],
                    params['s1x'], params['s1y'], params['s1z'],
                    params['s2x'], params['s2y'], params['s2z'],
                    params['lambda1'], params['lambda2'],
                    params['distance'], params['iota'] ,
                    params['f_min'], params['phi_ref'], 0.,
                    tidalorder=tidalorder,
                    usequadrupolemonopole=1)


def taylorf2_wrapper(freqs , params,
                     phaseorder = 11,
                     tidalorder = 15,
                     useqm = 1):
    """
        Wrapper for TF2 (generic)
        Defaults: 5.5PN + (3.5 PN QM) + (7.5PN tides) , (if Lam1 or Lam2 != 0)
    """
    return TaylorF2(phaseorder, freqs, params['mtot'], params['q'],
                    params['s1x'], params['s1y'], params['s1z'],
                    params['s2x'], params['s2y'], params['s2z'],
                    params['lambda1'], params['lambda2'],
                    params['distance'], params['iota'] ,
                    params['f_min'], params['phi_ref'], 0.,
                    tidalorder=tidalorder,
                    usequadrupolemonopole=useqm)
