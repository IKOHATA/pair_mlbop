// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Aidan Thompson (SNL) - original Tersoff implementation
                        Ikuma Kohata (The University of Tokyo)  - MLBOP implementation
------------------------------------------------------------------------- */

#include "pair_mlbop.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "math_special.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "suffix.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <array>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;
using namespace MathExtra;

#define DELTA 4

/* ---------------------------------------------------------------------- */

PairMLBOP::PairMLBOP(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

  params = nullptr;

  maxshort = 10;
  neighshort = nullptr;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairMLBOP::~PairMLBOP()
{
  if (copymode) return;

  memory->destroy(params);
  memory->destroy(elem3param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
  }
}

/* ---------------------------------------------------------------------- */

void PairMLBOP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  if (shift_flag) {
    if (evflag) {
      if (eflag) {
        if (vflag_either) eval<1,1,1,1>();
        else eval<1,1,1,0>();
      } else {
        if (vflag_either) eval<1,1,0,1>();
        else eval<1,1,0,0>();
      }
    } else eval<1,0,0,0>();

  } else {

    if (evflag) {
      if (eflag) {
        if (vflag_either) eval<0,1,1,1>();
        else eval<0,1,1,0>();
      } else {
        if (vflag_either) eval<0,1,0,1>();
        else eval<0,1,0,0>();
      }
    } else eval<0,0,0,0>();
  }
}

template <int SHIFT_FLAG, int EVFLAG, int EFLAG, int VFLAG_EITHER>
void PairMLBOP::eval()
{
  int i,j,k,ii,jj,kk,ll,inum,jnum;
  int itype,jtype,ktype,iparam_ij,iparam_ijk;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,evdwl1,evdwl2,fpair;
  double fforce1,fforce2;
  double rsq,rsq1,rsq2;
  double delr1[3],delr2[3],fi[3],fj[3],fk[3];
  double r1_hat[3],r2_hat[3];
  double partial_zeta[NL3], prefactoraij[NL3], prefactorbij[NL3], zeta_ij[NL3];
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  const double cutshortsq = cutmax*cutmax;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  // loop over full neighbor list of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      // shift rsq and store correction for force

      if (rsq < cutshortsq) {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }
    }

    // three-body interactions
    // skip immediately if I-J is not within cutoff
    double fjxtmp,fjytmp,fjztmp;

    for (jj = 0; jj < numshort; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      iparam_ij = elem3param[itype][jtype][jtype];

      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

     

      if (SHIFT_FLAG)
        rsq1 += shift*shift + 2*sqrt(rsq1)*shift;

      if (rsq1 >= params[iparam_ij].cutsq) continue;

      const double r1inv = 1.0/sqrt(dot3(delr1, delr1));
      scale3(r1inv, delr1, r1_hat);

      // accumulate bondorder zeta for each i-j interaction via loop over k
      
      fjxtmp = fjytmp = fjztmp = 0.0;
      for(ll=0;ll<NL3;ll++){zeta_ij[ll]=0;}
      //std::cout << i << j << "\n";
      for (kk = 0; kk < numshort; kk++) {

        if (jj == kk) continue;
        k = neighshort[kk];
        ktype = map[type[k]];
        iparam_ijk = elem3param[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

        if (SHIFT_FLAG)
          rsq2 += shift*shift + 2*sqrt(rsq2)*shift;

        if (rsq2 >= params[iparam_ijk].cutsq) continue;

        double r2inv = 1.0/sqrt(dot3(delr2, delr2));
        scale3(r2inv, delr2, r2_hat);

        calc_zeta(&params[iparam_ijk],rsq1,rsq2,r1_hat,r2_hat,partial_zeta);

        for(ll = 0; ll < NL3; ll++) {
          zeta_ij[ll] += partial_zeta[ll]; 
          //std::cout << partial_zeta[ll] << "\n";
        }
      }

      //std::cout << jj << "\n";

      // pairwise force due to zeta

      //force_zeta(&params[iparam_ij],rsq1,zeta_ij,fforce,prefactor,EFLAG,evdwl);
      force_aij(&params[iparam_ij],rsq1,zeta_ij,fforce1,prefactoraij,EFLAG,evdwl1);
      force_bij(&params[iparam_ij],rsq1,zeta_ij,fforce2,prefactorbij,EFLAG,evdwl2);


      fpair = (fforce1+fforce2)*r1inv;
      evdwl = evdwl1 + evdwl2;

      fxtmp += delr1[0]*fpair;
      fytmp += delr1[1]*fpair;
      fztmp += delr1[2]*fpair;
      fjxtmp -= delr1[0]*fpair;
      fjytmp -= delr1[1]*fpair;
      fjztmp -= delr1[2]*fpair;

      if (EVFLAG) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,0.0,-fpair,-delr1[0],-delr1[1],-delr1[2]);

      // attractive term via loop over k

      for (kk = 0; kk < numshort; kk++) {
        if (jj == kk) continue;
        k = neighshort[kk];
        ktype = map[type[k]];
        iparam_ijk = elem3param[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

        if (SHIFT_FLAG)
          rsq2 += shift*shift + 2*sqrt(rsq2)*shift;

        if (rsq2 >= params[iparam_ijk].cutsq) continue;

        double r2inv = 1.0/sqrt(dot3(delr2, delr2));
        scale3(r2inv, delr2, r2_hat);

        attractive(&params[iparam_ijk],prefactoraij,prefactorbij,
                   rsq1,rsq2,r1_hat,r2_hat,fi,fj,fk);

        fxtmp += fi[0];
        fytmp += fi[1];
        fztmp += fi[2];
        fjxtmp += fj[0];
        fjytmp += fj[1];
        fjztmp += fj[2];
        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];

        if (VFLAG_EITHER) v_tally3(i,j,k,fj,fk,delr1,delr2);
      }
      f[j][0] += fjxtmp;
      f[j][1] += fjytmp;
      f[j][2] += fjztmp;
    }
    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairMLBOP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMLBOP::settings(int narg, char **arg)
{

  // default values

  shift_flag = 0;

  // process optional keywords

  int iarg = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"shift") == 0) {
      if (suffix_flag & (Suffix::INTEL|Suffix::GPU|Suffix::KOKKOS))
        error->all(FLERR,"Keyword 'shift' not supported for this style");
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style command");
      shift = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      shift_flag = 1;
      iarg += 2;
    } else error->all(FLERR,"Illegal pair_style command");
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLBOP::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  map_element2type(narg-3,arg+3);

  // read potential file and initialize potential parameters

  read_file(arg[2]);
  setup_params();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairMLBOP::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Tersoff requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Tersoff requires newton pair on");

  // need a full neighbor list

  neighbor->add_request(this,NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMLBOP::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairMLBOP::read_file(char *file)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, file, "mlbop", unit_convert_flag);
    char *line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY,unit_convert);

    while ((line = reader.next_line(3))) {
      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();
        std::string kname = values.next_string();

        //std::cout << iname << " " << jname << " " << kname << "\n";

        // ielement,jelement,kelement = 1st args
        // if all 3 args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement, kelement;
        std::string ignore;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements){
          for (int i=0;i<NPARAMS_PER_TRIPLE;i++){reader.skip_line();}
          continue;
        } 
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements){ 
          for (int i=0;i<NPARAMS_PER_TRIPLE;i++){reader.skip_line();}
          continue;
        }
        for (kelement = 0; kelement < nelements; kelement++)
          if (kname == elements[kelement]) break;
        if (kelement == nelements){
          for (int i=0;i<NPARAMS_PER_TRIPLE;i++){reader.skip_line();}
          continue;
        }

        // load up parameter settings and error check their values

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params,maxparam*sizeof(Param), "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA*sizeof(Param));
        }

        params[nparams].ielement  = ielement;
        params[nparams].jelement  = jelement;
        params[nparams].kelement  = kelement;
        params[nparams].Rc = reader.next_double();
        params[nparams].Bc = reader.next_double();
        params[nparams].bo_Rc = reader.next_double();
        params[nparams].bo_Bc = reader.next_double();
        reader.next_dvector(params[nparams].L1, 3);
        reader.next_dvector(params[nparams].A, 3);
        params[nparams].Q = reader.next_double();
        reader.next_dvector(params[nparams].L2, 3);
        reader.next_dvector(params[nparams].B, 3);
        reader.next_dvector(params[nparams].l1weight, NL1);
        reader.next_dvector(params[nparams].l1bias, NL1);
        reader.next_dvector(params[nparams].l2weight, NL1);
        reader.next_dvector(params[nparams].l2bias, NL1);
        reader.next_dvector(params[nparams].l3weight, NL1);
        reader.next_dvector(params[nparams].l3bias, NL1);
        reader.next_dvector(params[nparams].l22weight, NL1*NL2);
        reader.next_dvector(params[nparams].l22bias, NL2);
        reader.next_dvector(params[nparams].l33weight, NL2*NL3);
        reader.next_dvector(params[nparams].l33bias, NL3);
        reader.next_dvector(params[nparams].al44weight, NL3*NL4);
        reader.next_dvector(params[nparams].al44bias, NL4);
        reader.next_dvector(params[nparams].al55weight, NL4*NL5);
        reader.next_dvector(params[nparams].al55bias, NL5);
        reader.next_dvector(params[nparams].al66weight, NL5*1);
        reader.next_dvector(params[nparams].al66bias, 1);
        params[nparams].aijbias = reader.next_double();
        reader.next_dvector(params[nparams].bl44weight, NL3*NL4);
        reader.next_dvector(params[nparams].bl44bias, NL4);
        reader.next_dvector(params[nparams].bl55weight, NL4*NL5);
        reader.next_dvector(params[nparams].bl55bias, NL5);
        reader.next_dvector(params[nparams].bl66weight, NL5*1);
        reader.next_dvector(params[nparams].bl66bias, 1);
        params[nparams].bijbias = reader.next_double();

        //std::cout << params[nparams].bijbias << "\n";
        

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params,maxparam*sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam*sizeof(Param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairMLBOP::setup_params()
{
  int i,j,k,m,n;

  // set elem3param for all element triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  memory->destroy(elem3param);
  memory->create(elem3param,nelements,nelements,nelements,"pair:elem3param");

  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++)
      for (k = 0; k < nelements; k++) {
        n = -1;
        for (m = 0; m < nparams; m++) {
          if (i == params[m].ielement && j == params[m].jelement &&
              k == params[m].kelement) {
            if (n >= 0) error->all(FLERR,"Potential file has a duplicate entry for: {} {} {}",
                                   elements[i], elements[j], elements[k]);
            n = m;
          }
        }
        if (n < 0) error->all(FLERR,"Potential file is missing an entry for: {} {} {}",
                              elements[i], elements[j], elements[k]);
        elem3param[i][j][k] = n;
      }


  // compute parameter values derived from inputs

  for (m = 0; m < nparams; m++) {
    params[m].cut = params[m].Rc;
    params[m].cutsq = params[m].cut*params[m].cut;
  }

  // set cutmax to max of all params

  cutmax = 0.0;
  for (m = 0; m < nparams; m++)
    if (params[m].cut > cutmax) cutmax = params[m].cut;
}

/* ---------------------------------------------------------------------- */

void PairMLBOP::calc_zeta(Param *param, double rsqij, double rsqik,
                         double *rij_hat, double *rik_hat, double *partial_zeta)
{
  double rij,rik,costheta,fctmp;

  rij = sqrt(rsqij);
  rik = sqrt(rsqik);
  costheta = dot3(rij_hat,rik_hat);
  
  double layer1[NL1] = {0};
  double layer2[NL2] = {0};
  double zeta_tmp[NL3]={0};

  for (int i=0;i<NL1;i++){
    layer1[i] = param->l1weight[i]*rij + param->l2weight[i]*rik + param->l3weight[i]*costheta + param->l1bias[i] + param->l2bias[i] + param->l3bias[i];
    layer1[i] = tanh(layer1[i]);
  }

  for (int i=0;i<NL2;i++){
    for (int j=0;j<NL1;j++){
      layer2[i] += layer1[j]*param->l22weight[NL1*i+j];
    }
    layer2[i] += param->l22bias[i];
    layer2[i] = tanh(layer2[i]);
  }

  for (int i=0;i<NL3;i++){
    for (int j=0;j<NL2;j++){
      zeta_tmp[i] += layer2[j]*param->l33weight[NL2*i+j];
    }
    zeta_tmp[i] += param->l33bias[i];
    zeta_tmp[i] = tanh(zeta_tmp[i]);
  }

  fctmp = ters_fc_bo(rik,param);

  for(int i=0;i<NL3;i++){
    partial_zeta[i] = fctmp*zeta_tmp[i];
  }
}

/* ---------------------------------------------------------------------- */

void PairMLBOP::force_aij(Param *param, double rsq, double *zeta_ij,
                             double &fforce, double *prefactoraij,
                             int eflag, double &eng)
{
  int i,j,k;
  double r,fr,fr_d,tmpfr;
  double aij;
  double dbijdzeta[NL3] = {0};
  double tmp,tmpsq,tmpmat;
  double layer1[NL4] = {0};
  double layer2[NL5] = {0};
  double layer1_d[NL3*NL4] = {0};
  double layer2_d[NL4*NL5] = {0};
  double layer3_d[NL5*1] = {0};

  r = sqrt(rsq);
  fr = ters_fr(r,param);
  fr_d = ters_fr_d(r,param);

  
  

  for (i=0;i<NL4;i++){
    tmp = 0;
    for (j=0;j<NL3;j++){
      tmp += zeta_ij[j] * param->al44weight[NL3*i+j];
    }
    tmp += param->al44bias[i];
    tmp = tanh(tmp);
    layer1[i] = tmp;
    tmpsq = 1-tmp*tmp;
    for (j=0;j<NL3;j++){
      layer1_d[NL3*i+j] = tmpsq * param->al44weight[NL3*i+j];
    }
  }

  //std::cout << fr << "\n";

  for (i=0;i<NL5;i++){
    tmp = 0;
    for (j=0;j<NL4;j++){
      tmp += layer1[j] * param->al55weight[NL4*i+j];
    }
    tmp += param->al55bias[i];
    tmp = tanh(tmp);
    layer2[i] = tmp;
    tmpsq = 1-tmp*tmp;
    for (j=0;j<NL4;j++){
      layer2_d[NL4*i+j] = tmpsq * param->al55weight[NL4*i+j];
    }
  }

  

  tmp = 0;
  for (j=0;j<NL5;j++){
    tmp += layer2[j] * param->al66weight[j];
  }
  tmp += param->al66bias[0]- param->aijbias;
  aij = (1/loge2)*log(1+exp(loge2*tmp));
  tmpsq = 1/(1+exp(-loge2*tmp));
  for (j=0;j<NL5;j++){
    layer3_d[j] = tmpsq * param->al66weight[j];
  }

  for (i=0;i<NL5;i++){
    for (j=0;j<NL4;j++){
      tmpmat = layer2_d[NL4*i+j] * layer3_d[i];
      for (k=0;k<NL3;k++){
        dbijdzeta[k] += layer1_d[NL3*j+k] * tmpmat;
      }
    }
  }

  
  //std::cout << bij << "\n";
  fforce = -0.5*aij*fr_d;
  //prefactor = -0.5*fa*ters_bij_d20(zeta_ijj,param);
  tmpfr = -0.5*fr;
  for (i=0;i<NL3;i++){prefactoraij[i] = tmpfr*dbijdzeta[i];}
  if (eflag) eng = 0.5*aij*fr;
}

/* ---------------------------------------------------------------------- */

void PairMLBOP::force_bij(Param *param, double rsq, double *zeta_ij,
                             double &fforce, double *prefactorbij,
                             int eflag, double &eng)
{
  int i,j,k;
  double r,fa,fa_d,tmpfa;
  double bij;
  double dbijdzeta[NL3] = {0};
  double tmp,tmpsq,tmpmat;
  double layer1[NL4] = {0};
  double layer2[NL5] = {0};
  double layer1_d[NL3*NL4] = {0};
  double layer2_d[NL4*NL5] = {0};
  double layer3_d[NL5*1] = {0};

  r = sqrt(rsq);
  fa = ters_fa(r,param);
  fa_d = ters_fa_d(r,param);
  

  for (i=0;i<NL4;i++){
    tmp = 0;
    for (j=0;j<NL3;j++){
      tmp += zeta_ij[j] * param->bl44weight[NL3*i+j];
    }
    tmp += param->bl44bias[i];
    tmp = tanh(tmp);
    layer1[i] = tmp;
    tmpsq = 1-tmp*tmp;
    for (j=0;j<NL3;j++){
      layer1_d[NL3*i+j] = tmpsq * param->bl44weight[NL3*i+j];
    }
  }

  for (i=0;i<NL5;i++){
    tmp = 0;
    for (j=0;j<NL4;j++){
      tmp += layer1[j] * param->bl55weight[NL4*i+j];
    }
    tmp += param->bl55bias[i];
    tmp = tanh(tmp);
    layer2[i] = tmp;
    tmpsq = 1-tmp*tmp;
    for (j=0;j<NL4;j++){
      layer2_d[NL4*i+j] = tmpsq * param->bl55weight[NL4*i+j];
    }
  }

  tmp = 0;
  for (j=0;j<NL5;j++){
    tmp += layer2[j] * param->bl66weight[j];
  }
  tmp += param->bl66bias[0]- param->bijbias;
  bij = (1/loge2)*log(1+exp(loge2*tmp));
  tmpsq = 1/(1+exp(-loge2*tmp));
  for (j=0;j<NL5;j++){
    layer3_d[j] = tmpsq * param->bl66weight[j];
  }

  for (i=0;i<NL5;i++){
    for (j=0;j<NL4;j++){
      tmpmat = layer2_d[NL4*i+j] * layer3_d[i];
      for (k=0;k<NL3;k++){
        dbijdzeta[k] += layer1_d[NL3*j+k] * tmpmat;
      }
    }
  }
  //std::cout << bij << "\n";
  fforce = 0.5*bij*fa_d;
  //prefactor = -0.5*fa*ters_bij_d20(zeta_ijj,param);
  tmpfa = -0.5*fa;
  for (i=0;i<NL3;i++){prefactorbij[i] = tmpfa*dbijdzeta[i];}
  if (eflag) eng = 0.5*bij*fa;
}

/* ----------------------------------------------------------------------
   attractive term
   use param_ij cutoff for rij test
   use param_ijk cutoff for rik test
------------------------------------------------------------------------- */

void PairMLBOP::attractive(Param *param, double *prefactoraij, double *prefactorbij,
                             double rsqij, double rsqik,
                             double *rij_hat, double *rik_hat,
                             double *fi, double *fj, double *fk)
{
  double rij,rijinv,rik,rikinv;

  rij = sqrt(rsqij);
  rik = sqrt(rsqik);

  rijinv = 1.0/rij;
  rikinv = 1.0/rik;

  ters_zetaterm_d(prefactoraij,prefactorbij,rij_hat,rij,rijinv,rik_hat,rik,rikinv,fi,fj,fk,param);
}

/* ---------------------------------------------------------------------- */

double PairMLBOP::ters_fc(double r, Param *param)
{
  double ters_R = param->Rc;
  double ters_B = param->Bc;

  if (r > ters_R) return 0.0;
  return cube(tanh((1.0-r/ters_R)*ters_B)/tanh(ters_B));
}

/* ---------------------------------------------------------------------- */

double PairMLBOP::ters_fc_d(double r, Param *param)
{
  double ters_R = param->Rc;
  double ters_B = param->Bc;

  if (r > ters_R) return 0.0;
  double tanh2 = square(tanh((1.0-r/ters_R)*ters_B));
  return 3.0*(-ters_B/ters_R)*cube(1/tanh(ters_B))*tanh2*(1.0-tanh2);
}

/* ---------------------------------------------------------------------- */

double PairMLBOP::ters_fc_bo(double r, Param *param)
{
  double ters_R = param->bo_Rc;
  double ters_B = param->bo_Bc;

  if (r > ters_R) return 0.0;
  return cube(tanh((1.0-r/ters_R)*ters_B)/tanh(ters_B));
}

/* ---------------------------------------------------------------------- */

double PairMLBOP::ters_fc_bo_d(double r, Param *param)
{
  double ters_R = param->bo_Rc;
  double ters_B = param->bo_Bc;

  if (r > ters_R) return 0.0;
  double tanh2 = square(tanh((1.0-r/ters_R)*ters_B));
  return 3.0*(-ters_B/ters_R)*cube(1/tanh(ters_B))*tanh2*(1.0-tanh2);
}

/* ---------------------------------------------------------------------- */

double PairMLBOP::ters_fr(double r, Param *param)
{ 
  double tmp_exp;
  if (r > params->Rc) return 0.0;
  tmp_exp = 0;

  for (int i=0;i<3;i++){
    tmp_exp += param->A[i] * exp(-param->L1[i] * r);
  }

  return ters_fc(r,param) * (1.0 + param->Q/r) * tmp_exp;
}

/* ---------------------------------------------------------------------- */

double PairMLBOP::ters_fr_d(double r, Param *param)
{
  double tmp,tmp_exp,tmp_exp_d,tmp_eng,tmp_fc,tmp_fc_d;
  if (r > params->Rc) return 0.0;
  tmp_exp = 0;
  tmp_exp_d = 0;
  tmp_fc = ters_fc(r,param);
  tmp_fc_d = ters_fc_d(r,param);

  for (int i=0;i<3;i++){
    tmp = param->A[i] * exp(-param->L1[i] * r);
    tmp_exp += tmp;
    tmp_exp_d += -param->L1[i] * tmp;
  }

  tmp_eng = tmp_fc * tmp_exp;

  return - ( (1.0 + param->Q/r) * (tmp_exp * tmp_fc_d + tmp_fc * tmp_exp_d) + (-param->Q/square(r)) * tmp_eng  );
}

/* ---------------------------------------------------------------------- */

double PairMLBOP::ters_fa(double r, Param *param)
{ 
  double tmp_exp;
  if (r > params->Rc) return 0.0;
  tmp_exp = 0;

  for (int i=0;i<3;i++){
    tmp_exp += param->B[i] * exp(-param->L2[i] * r);
  }

  return - tmp_exp * ters_fc(r,param);
}

/* ---------------------------------------------------------------------- */

double PairMLBOP::ters_fa_d(double r, Param *param)
{
  double tmp,tmp_exp,tmp_exp_d;
  if (r > params->Rc) return 0.0;
  tmp_exp = 0;
  tmp_exp_d = 0;

  for (int i=0;i<3;i++){
    tmp = param->B[i] * exp(-param->L2[i] * r);
    tmp_exp += tmp;
    tmp_exp_d += -param->L2[i] * tmp;
  }

  return - (tmp_exp * ters_fc_d(r,param) + ters_fc(r,param) * tmp_exp_d);
}

/* ---------------------------------------------------------------------- */

void PairMLBOP::ters_zetaterm_d(double *prefactoraij, double *prefactorbij,
                                  double *rij_hat, double rij, double rijinv,
                                  double *rik_hat, double rik, double rikinv,
                                  double *dri, double *drj, double *drk,
                                  Param *param)
{
  int i,j,k;
  double fc,dfc,cos_theta,tmp,tmpsq,tmpmat1,tmpmat2,tmpmat3;
  double dcosdri[3],dcosdrj[3],dcosdrk[3];
  double zeta[NL3] = {0};
  double dzetadrij[NL3] = {0};
  double dzetadrik[NL3] = {0};
  double dzetadcos[NL3] = {0};

  double layer1[NL1] = {0};
  double layer2[NL2] = {0};
  double layer1_d[3*NL1] = {0};
  //layer1_d[0*5+i] -> drij, layer1_d[1*5+i] -> drik, layer1_d[2*5+i] -> dcos
  double layer2_d[NL1*NL2] = {0};
  double layer3_d[NL2*NL3] = {0};
  
  double fabij = 0;
  double fadbijdrij = 0;
  double fadbijdrik = 0;
  double fadbijdcos = 0;

  fc = ters_fc_bo(rik,param);
  dfc = ters_fc_bo_d(rik,param);
  cos_theta = dot3(rij_hat,rik_hat);

  for (i=0;i<NL1;i++){
      tmp = param->l1weight[i]*rij + param->l2weight[i]*rik + param->l3weight[i]*cos_theta + param->l1bias[i] + param->l2bias[i] + param->l3bias[i];
      tmp = tanh(tmp);
      layer1[i] = tmp;
      tmpsq = 1-tmp*tmp;
      layer1_d[i] = tmpsq * param->l1weight[i];
      layer1_d[NL1+i] = tmpsq * param->l2weight[i];
      layer1_d[NL1*2+i] = tmpsq * param->l3weight[i];
    }

  for (i=0;i<NL2;i++){
    tmp = 0;
    for (j=0;j<NL1;j++){
      tmp += layer1[j]*param->l22weight[NL1*i+j];
    }
    tmp += param->l22bias[i];
    tmp = tanh(tmp);
    layer2[i] = tmp;
    tmpsq = 1-tmp*tmp;
    for (j=0;j<NL1;j++){
      layer2_d[NL1*i+j] = tmpsq * param->l22weight[NL1*i+j];
    }
  }

  for (i=0;i<NL3;i++){
    tmp = 0;
    for (j=0;j<NL2;j++){
      tmp += layer2[j]*param->l33weight[NL2*i+j];
    }
    tmp += param->l33bias[i];
    tmp = tanh(tmp);
    zeta[i] = tmp;
    tmpsq = 1-tmp*tmp;
    for (j=0;j<NL2;j++){
      layer3_d[NL2*i+j] = tmpsq * param->l33weight[NL2*i+j];
    }
  }

  for (i=0;i<NL2;i++){
    for (j=0;j<NL1;j++){
        tmpmat1 = layer1_d[j] * layer2_d[NL1*i+j];
        tmpmat2 = layer1_d[NL1+j] * layer2_d[NL1*i+j];
        tmpmat3 = layer1_d[NL1*2+j] * layer2_d[NL1*i+j];
      for (k=0;k<NL3;k++){
        dzetadrij[k] += layer3_d[NL2*k+i] * tmpmat1;
        dzetadrik[k] += layer3_d[NL2*k+i] * tmpmat2;
        dzetadcos[k] += layer3_d[NL2*k+i] * tmpmat3;
      }
    }
  }
  
  for (i=0;i<NL3;++i){
    fabij += (prefactoraij[i] + prefactorbij[i]) * zeta[i];
    fadbijdrij += (prefactoraij[i] + prefactorbij[i]) * dzetadrij[i];
    fadbijdrik += (prefactoraij[i] + prefactorbij[i]) * dzetadrik[i];
    fadbijdcos += (prefactoraij[i] + prefactorbij[i]) * dzetadcos[i];
  }

  costheta_d(rij_hat,rijinv,rik_hat,rikinv,dcosdri,dcosdrj,dcosdrk);

  // compute the derivative wrt Ri
  // dri = -dfc*gijk*ex_delr*rik_hat;
  // dri += fc*gijk_d*ex_delr*dcosdri;
  // dri += fc*gijk*ex_delr_d*(rik_hat - rij_hat);

  scale3(-dfc*fabij,rik_hat,dri);
  scaleadd3(fc*fadbijdcos,dcosdri,dri,dri);
  scaleadd3(-fc*fadbijdrij,rij_hat,dri,dri);
  scaleadd3(-fc*fadbijdrik,rik_hat,dri,dri);
  

  // compute the derivative wrt Rj
  // drj = fc*gijk_d*ex_delr*dcosdrj;
  // drj += fc*gijk*ex_delr_d*rij_hat;

  scale3(fc*fadbijdcos,dcosdrj,drj);
  scaleadd3(fc*fadbijdrij,rij_hat,drj,drj);

  // compute the derivative wrt Rk
  // drk = dfc*gijk*ex_delr*rik_hat;
  // drk += fc*gijk_d*ex_delr*dcosdrk;
  // drk += -fc*gijk*ex_delr_d*rik_hat;

  scale3(dfc*fabij,rik_hat,drk);
  scaleadd3(fc*fadbijdcos,dcosdrk,drk,drk);
  scaleadd3(fc*fadbijdrik,rik_hat,drk,drk);
}

/* ---------------------------------------------------------------------- */
void PairMLBOP::costheta_d(double *rij_hat, double rijinv,
                             double *rik_hat, double rikinv,
                             double *dri, double *drj, double *drk)
{
  // first element is devative wrt Ri, second wrt Rj, third wrt Rk

  double cos_theta = dot3(rij_hat,rik_hat);

  scaleadd3(-cos_theta,rij_hat,rik_hat,drj);
  scale3(rijinv,drj);
  scaleadd3(-cos_theta,rik_hat,rij_hat,drk);
  scale3(rikinv,drk,drk);
  add3(drj,drk,dri);
  scale3(-1.0,dri);
}

/* ---------------------------------------------------------------------- */