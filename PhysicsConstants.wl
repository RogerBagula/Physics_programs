(* ::Package:: *)

(* ::Package:: *)

BeginPackage["PhysicsConstants`"];

N0::usage = "Avogadro constant.";
G::usage = "Gravitational constant.";
e::usage = "Electron charge.";
c::usage = "Speed of light.";
hbar::usage = "Reduced Planck constant.";
me::usage = "Electron mass.";
mev::usage = "Electron mass in MeV.";
mpi::usage = "Pion mass.";
tpo::usage = "Positron lifetime.";
rb::usage = "Bohr radius.";
Qm::usage = "Monopole charge.";

ml::usage = "Lepton masses.";
Ql::usage = "Lepton volume charges.";

mln::usage = "Neutrino masses.";
Qln::usage = "Neutrino volume charges.";

alpha::usage = "Fine structure constant.";
alpha0::usage = "Low-energy alpha.";
alphaw::usage = "Weak alpha.";
Gf::usage = "Fermi constant.";
Omega::usage = "Weak mixing parameter.";
kb::usage = "Boltzmann constant.";

mH::usage = "Higgs mass.";
mp::usage = "Proton mass.";
mn::usage = "Neutron mass.";

thetaW0::usage = "Weinberg angle.";
thetaWrad::usage = "Weinberg angle (radians).";
thetaWdeg::usage = "Weinberg angle (degrees).";

g::usage = "Electron g-factor.";
muni::usage = "Unified mass (AdS model).";

mhiggs::usage = "Higgs mass.";
mz::usage = "Z boson mass.";
mw::usage = "W boson mass.";
mtop::usage = "Top quark mass.";
mbottom::usage = "Bottom quark mass.";
mcharm::usage = "Charm quark mass.";
mstrange::usage = "Strange quark mass.";

mplanck::usage = "Planck mass.";

Tvac::usage = "Vacuum temperature.";
ru::usage = "Observable universe radius.";
rplanck::usage = "Planck length.";
tplanck::usage = "Planck time.";
tu::usage = "Universe age.";
tun::usage = "New universe age.";
run::usage = "New universe radius.";

Cw::usage = "Weak charge on 5D simplex.";
rw::usage = "Weak radius.";

vw::usage = "Weeks volume.";
vt::usage = "Thurston-Meyerhoff volume.";
vl::usage = "Lepton hyperbolic 3-manifold volume.";

ab::usage = "Bohr radius (precise).";
tpip::usage = "Charged pion lifetime.";
tpiz::usage = "Neutral pion lifetime.";

mqmu::usage = "Muon quark product mass.";
mqtau::usage = "Tau quark product mass.";
mqs::usage = "Strange quark product mass.";
mqc::usage = "Charm quark product mass.";

ttau::usage = "Tau lifetime.";
tmu::usage = "Muon lifetime.";

Hcbm::usage = "Hubble constant (CBM).";
Hsnova::usage = "Hubble constant (supernova).";
Hlense::usage = "Hubble constant (lensing).";
Have::usage = "Average Hubble constant.";
H0::usage = "Hubble constant.";
th0::usage = "Hubble time.";

tcmb::usage = "CMB age.";
msun::usage = "Solar mass.";

g0::usage = "Single-loop GED weak g'.";
g1::usage = "Weak g1.";

gamma::usage = "Gravity moment.";
munified::usage = "Unified mass.";

Begin["`Private`"];

N0 = 6.02214076*10^23;
G = 6.67259*10^-8;
e = 4.8032068*10^-10;
c = 2.99792458*10^10;
hbar = 1.05457266*10^-27;
me = 9.10938291*10^-28;
mev = 0.510998928;
mpi = 139.57018*me/mev;

tpo = 0.4787*10^-9;
rb = 5.29*10^-9;
Qm = 2*Pi*hbar/e;

mtau = 1776.82*me/mev;
mmu  = 105.6583715*me/mev;
ml   = {me, mmu, mtau};
Ql   = {0.993708, 5.87607, 15.0545};

mnu    = 1.78266270*10^-33;
mnumu  = 6.7978246932973695*^-31;
mnutau = 1.1431743145263108*^-29;
mln    = {mnu, mnumu, mnutau};
Qln    = {0.0124295, 0.0901334, 0.230923};

alpha  = 1/137.035999206;
alpha0 = 0.007296628362685409;
alphaw = 0.42541281596347497;
Gf     = 1.4327038609285911*^-49;
Omega  = 0.0078749969978123844;
kb     = 1.380658*10^-16;

mH = 126.22136632802759*10^3*me/mev;
mp = 1.672621777*10^-24;
mn = 1838.6402*me;

thetaW0   = 0.231208;
thetaWrad = ArcSin[Sqrt[thetaW0]];
thetaWdeg = thetaWrad*180/Pi;

g    = 2*1.00115965218085;
muni = 3.6543166832631125*^-12;

mhiggs  = 125.38*10^3*me/mev;
mz      = 91.1876*10^3*me/mev;
mw      = 80.4335*10^3*me/mev;
mtop    = 172.93*10^3*me/mev;
mbottom = 4.1*10^3*me/mev;
mcharm  = 1.275*10^3*me/mev;
mstrange = 92.4*me/mev;

mplanck = Sqrt[hbar*c/G];

Tvac    = 2.72548;
ru      = 8.8*10^28/2;
rplanck = 2*Pi*hbar/(mplanck*c);
tplanck = rplanck/c;

tu  = 13.798*10^9*365.25*24*60*60;
tun = 26.7*10^9*365.25*24*60*60;
run = c*tun;

Cw = (3!*2!)*Det[{
   {alpha, 0, 0, 0, 0, 1},
   {0, alpha, 0, 0, 0, 1},
   {0, 0, alpha, 0, 0, 1},
   {0, 0, 0, alpha, 0, 1},
   {0, 0, 0, 0, alpha, 1},
   {0, 0, 0, 0, 0, 1}
   }]/5!;
rw = Cw^2/(mnu*c^2);

vw = 0.94270736277692772092;
vt = 0.981368828892232088914;
vl = (18/24)*vt;

ab   = 5.2917721067*10^-9;
tpip = 2.6033*10^-8;
tpiz = 8.4*10^-17;

mqmu  = Sqrt[mmu/me];
mqtau = Sqrt[mtau/me];
mqs   = mqtau*mqmu;
mqc   = 2*mqs;

ttau = 2.932*10^-13;
tmu  = 2.1969811*10^-6;

Hcbm   = 67.4/(3.09*10^19);
Hsnova = 74.0/(3.09*10^19);
Hlense = 73.3/(3.09*10^19);
Have   = 73.8/(3.09*10^19);
H0     = 2.2894835758225842*^-18;
th0    = 1/H0;

tcmb = 372000*365.25*24*60*60;
msun = 1.9891*10^33;

g0 = 2*(1 + alpha/2);
g1 =g1 = 0.7760034784132269;

gamma    = e*G/(hbar*c);
munified = hbar/(2*alpha*c);

End[];

EndPackage[];

