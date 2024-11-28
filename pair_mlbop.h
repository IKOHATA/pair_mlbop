/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mlbop,PairMLBOP);
// clang-format on
#else

#ifndef LMP_PAIR_MLBOP_H
#define LMP_PAIR_MLBOP_H

#include "pair.h"

#include <array>

#define NL1 5
#define NL2 10
#define NL3 20
#define NL4 20
#define NL5 20

namespace LAMMPS_NS {

class PairMLBOP : public Pair {
 public:
  PairMLBOP(class LAMMPS *);
  ~PairMLBOP() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  template <int SHIFT_FLAG, int EVFLAG, int EFLAG, int VFLAG_ATOM> void eval();

  static constexpr int NPARAMS_PER_TRIPLE = 7 + 6 + NL2+1 + NL3+1 + NL4+1 + NL5+1 + 2; 
  double loge2 = log(2);

  struct Param {
    double Rc,Bc;
    double bo_Rc,bo_Bc;
    double Q;
    double L1[3],A[3];
    double L2[3],B[3];
    double l1weight[NL1],l1bias[NL1];
    double l2weight[NL1],l2bias[NL1];
    double l3weight[NL1],l3bias[NL1];
    double l22weight[NL1*NL2],l22bias[NL2];
    double l33weight[NL2*NL3],l33bias[NL3];
    double al44weight[NL3*NL4],al44bias[NL4];
    double al55weight[NL4*NL5],al55bias[NL5];
    double al66weight[NL5*1],al66bias[1];
    double aijbias;
    double bl44weight[NL3*NL4],bl44bias[NL4];
    double bl55weight[NL4*NL5],bl55bias[NL5];
    double bl66weight[NL5*1],bl66bias[1];
    double bijbias;
    double cut, cutsq;
    int ielement, jelement, kelement;
  };

 protected:
  Param *params;      // parameter set for an I-J-K interaction
  double cutmax;      // max cutoff for all elements
  int maxshort;       // size of short neighbor list array
  int *neighshort;    // short neighbor list array

  int shift_flag;    // flag to turn on/off shift
  double shift;      // negative change in equilibrium bond length

  virtual void allocate();
  virtual void read_file(char *);
  virtual void setup_params();
  //virtual void repulsive(Param *, double, double &, int, double &);
  //virtual std::array<double, 20> zeta(Param *, double, double, double *, double *);
  virtual void calc_zeta(Param *, double, double, double *, double *, double *);
  //virtual void force_zeta(Param *, double, std::array<double, 20>, double &, std::array<double, 20> &, int, double &);
  virtual void force_aij(Param *, double, double *, double &, double *, int, double &);
  virtual void force_bij(Param *, double, double *, double &, double *, int, double &);
  void attractive(Param *, double *, double *, double, double, double *, double *, double *, double *,
                  double *);
  
  virtual double ters_fc(double, Param *);
  virtual double ters_fc_d(double, Param *);
  virtual double ters_fc_bo(double, Param *);
  virtual double ters_fc_bo_d(double, Param *);
  virtual double ters_fr(double, Param *);
  virtual double ters_fr_d(double, Param *);
  virtual double ters_fa(double, Param *);
  virtual double ters_fa_d(double, Param *);

  virtual void ters_zetaterm_d(double *, double *, double *, double, double, double *, double, double,
                               double *, double *, double *, Param *);
  void costheta_d(double *, double, double *, double, double *, double *, double *);

};

}    // namespace LAMMPS_NS

#endif
#endif
