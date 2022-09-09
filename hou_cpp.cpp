// to compile on Kepler: g++ -std=c++11 hou_cpp.cpp -o hou_cpp_final -I /Users/alda8226/binary_asteroids/armadillo-8.300.4/include -framework Accelerate
// to compile on Nexus: /bin/g++ -std=c++11 hou_cpp.cpp -o hou_cpp_final -larmadillo
// -larmadillo ensures the Armadill library (http://arma.sourceforge.net/docs.html) can be found during complilation
//
// Hou_cpp.cpp is the .cpp file used to compile the mutual gravity, dynamics and integration executable "hou_cpp_final"
// which is used to generate ephemeris for user defined binary systems. In order for it to run the user must use C++11
// and the Armadillo library (link provided at top of file). The Armadillo library gives C++ matrix functionality and allows
// for quick evaluation of the dynamics. The mutual potential and related functions are develloped from Hou 2016 and 
// documentation assumes the reader is familiar with the paper. Many of the functions herein also exist in the provided python
// scripts and comparison may help in understanding the various functions. Please see the documentation for further details.
//
//
#define _USE_MATH_DEFINES
#include <iostream>//cout cin
#include <string>//string
#include <istream>//file io
#include <armadillo>//matrices etc.
#include <cmath>//math
using namespace std;
using namespace arma;

// structure storing pointers for  general system parameters
struct parameters{
    double* G;//gravity constant - units of km, kg
    cube* TA;//primary inertia integrals - units of km, kg
    cube* TB;//secondary inertia integrals - units of km,kg
    cube* TBp;//TB rotated from B to A frame
    cube* TS;// sphere inertia integral
    cube* Tsun;
    field<cube>* dT;//partials of TBp with respect to elements of the rotation matrix which transforms from to B to A
    mat* IA;//primary moments of inertia - kg, km
    mat* IB;//secondary moments of inertia - kg,km
    mat* IdA;//primary nonstandard moments of inertia used for LGVI - kg, km
    mat* IdB;//secondary nonstandard moments of inertia used for LGVI - kg, km
    double* m;// mass ratio(Mc*Ms/(Mc+Ms)) - km
    double* nu;// mass ratio(Ms/(Mc_MS)) å
    mat* tk;// set of tk expansion coefficients
    sp_mat* a;// set of a expansion coefficients
    sp_mat* b;// set of b expansion coefficients
    int* n;//mutual potential truncation order
    int* flyby_toggle;//toggle to turn on 3rd body perturber
    int* helio_toggle;//toggle to turn on solar perturbation
    int* sg_toggle;//toggle to turn on legacy solar perturbation
    int* tt_toggle;//toggle to turn on tidal torque
    double* Mplanet;//mass of perturbing 3rd body
    double* a_hyp;//semimajor axis of perturbing 3rd body
    double* e_hyp;//eccentricity of perturbing 3rd body
    double* i_hyp;//inclination of perturbing 3rd body
    double* RAAN_hyp;//RAAN of perturbing 3rd body
    double* om_hyp;//argument of periapsis of perturbing 3rd body
    double* tau_hyp;//time of periapsis passage of perturbing 3rd body
    double* n_hyp;//mean motion of perturbing 3rd body
    double* Msolar;//solar mass
    double* a_helio;//semimajor axis of solar orbit
    double* e_helio;//eccentricity of solar orbit
    double* i_helio;//inclination of solar orbit
    double* RAAN_helio;//RAAN of solar orbit
    double* om_helio;//argument of periapsis of solar orbit
    double* tau_helio;//time of periapsis passage of solar orbit
    double* n_helio;//mean motion of solar orbit
    double* sol_rad;//legacy radius of circular orbit about the sun
    double* au_def;//legacy definition of AU in km
    double* love1;//love number of the primary
    double* love2;//love number of the secondary
    double* refrad1;//primary reference radius
    double* refrad2;//secondary reference radius
    double* rhoA;//primary density
    double* rhoB;//secondary density
    double* eps1;//primary tidal lag angle (related to quality factor Q)
    double* eps2;//secondasry tidal lag angle (related to quality factor Q)
    double* mean_motion;//legacy mean motion of circular orbit about the sun
    mat* sg_acc;//legacy acceleration due to solar gravity
    mat* acc_3BP;//acceleration due to 3rd body perturber
    mat* acc_solar;//acceleration due to heliocentric orbit
    mat* tt_1;//tidal torque on the primary caused by the secondary
    mat* tt_2;//tidal torque on the secondary causecd by the primary
    mat* tt_orbit;//acceleration on the orbit due to tidal torque
} inputs;

// structure used to take in inputs file ("ic_inputs.txt")
struct initialization{
    double G;//gravity constant - units of km, kg
    int order;//mutual potential truncation order
    int order_a;//primary inertia integral truncation order
    int order_b;//secondary inertia integral truncation order
    double aA;//primary smja - m
    double bA;//primary sia - m
    double cA;//primary smna - m
    double aB;//secondary smja -m
    double bB;//secondary sia - m
    double cB;//secondary smna - m
    int a_shape;//primary shape type flag
    int b_shape;//secondary shape type flag
    double rhoA;//primary density - kg/km3
    double rhoB;//secondary density - kg/km3
    double t0;//initial time - sec
    double tf;//final time - sec
    string TAfile;//primary inertia integral file - not implemented
    string TBfile;//secondary inertia integral file - not implemented
    string IAfile;//primary inertia moment file - not implemented
    string IBfile;//secondary inertia moment file - not implemented
    string tfa;//primary tet file
    string vfa;//primary vert file
    string tfb;//secondary tet file
    string vfb;//secondary vert file
    mat x0;//initial conditions [r,v,wc,ws,Cc,C]
    int Tgen;//inertia integral generation flag
    int integ;//integrator choice flag
    double h;//fixed time step - sec
    double tol;//adaptive tolerance for rk78
    int flyby_toggle;//toggle to turn on legacy solar perturbation
    int helio_toggle;//toggle to turn on solar perturbation
    int sg_toggle;//toggle to turn on legacy solar perturbation
    int tt_toggle;//toggle to turn on tidal torque
    double Mplanet;//mass of perturbing 3rd body
    double a_hyp;//semimajor axis of perturbing 3rd body
    double e_hyp;//eccentricity of perturbing 3rd body
    double i_hyp;//inclination of perturbing 3rd body
    double RAAN_hyp;//RAAN of perturbing 3rd body
    double om_hyp;//argument of periapsis of perturbing 3rd body
    double tau_hyp;//time of periapsis passage of perturbing 3rd body
    double Msolar;//solar mass
    double a_helio;//semimajor axis of solar orbit
    double e_helio;//eccentricity of solar orbit
    double i_helio;//inclination of solar orbit
    double RAAN_helio;//RAAN of solar orbit
    double om_helio;//argument of periapsis of solar orbit
    double tau_helio;//time of periapsis passage of solar orbit
    double sol_rad;//legacy radius of circular orbit about the sun
    double au_def;//legacy definition of AU in km
    double love1;//love number of the primary
    double love2;//love number of the secondary
    double refrad1;//primary reference radius
    double refrad2;//secondary reference radius
    double eps1;//primary tidal lag angle (related to quality factor Q)
    double eps2;//secondasry tidal lag angle (related to quality factor Q)
    double Msun;//solar mass
} ics;

//function declarations - descriptions at file definitions
void tk_calc(int m, mat* t);
void a_calc(int n, sp_mat* a);
void b_calc(int n, sp_mat* b);
int t_ind(int a, int b, int c, int d, int e, int f, int g, int dim);
double Q_ijk(double i, double j, double k);
double tet_sums(double l, double m, double n, double x1, double x2, double x3, double y1, double y2, double y3, double z1, double z2, double z3);
void poly_inertia(int q, double rho, string tet_file, string vert_file, cube* T);
void poly_inertia_met(int q, double rho, string tet_file, string vert_file, cube* T);
void inertia_rot(mat C, int q, cube* T, cube* Tp);
void poly_moi(double rho, string tet_file, string vert_file, mat* I);
void poly_moi_met(double rho, string tet_file, string vert_file, mat* I);
void ell_mass_params_met(double order, double order_body,double rho, double a, double b, double c, mat* I, cube* T);
double u_tilde(int dim, int n, mat* t, sp_mat* a, sp_mat* b, mat e, cube* TA, cube* TBp);
double du_dx_tilde(int dim,int n, mat* t, sp_mat* a, sp_mat* b, mat e, double R, int dx, cube* TA, cube* TBp);
double de_dx(mat e, double R, int de, int dx);
void dT_dc(int i, int j, mat C, int q, cube* TA, cube* dT);
double du_dc_tilde(int dim,int n, mat* t, sp_mat* a, sp_mat* b, mat e, cube* TA, cube* dT);
double du_x(double G, int m, mat* t, sp_mat* a, sp_mat* b, mat e, double R, int dx, cube* TA, cube* TBp);
double du_c(double G, int m, mat* t, sp_mat* a, sp_mat* b, mat e, double R, cube* TA, cube* dT);
double potential(double G, int m, mat* t, sp_mat* a, sp_mat* b, mat e, double R, cube* TA, cube* TBp);
void rk87(double t0, double tf, mat x0, double rel_tol, parameters inputs, mat* tout, mat* xout, std::function<mat(mat,mat, parameters)> ode);
void rk4_stack(double t0, double tf, mat x0, double h, parameters inputs, mat* tout, mat* xout, mat* hout, mat* sunout, std::function<mat(mat,mat, parameters)> ode);
void ABM(double t0, double tf, mat x0, double h, parameters inputs, mat* tout, mat* xout, mat* hout, mat*sunout, std::function<mat(mat,mat, parameters)> ode);
void LGVI_integ(double t0, double tf, mat x0, double h, parameters inputs, mat* tout, mat* xout);
void hamiltonian_map(double h, parameters inputs, mat* x, mat* x_out, mat* fab, mat* fna, mat* Fab,
 mat* Fna, mat* g_map, mat* G_map, mat* grad_G, mat* du_dr_n, mat* M_n, mat* x0);
void F_cayley_calc(double h, mat* g, mat* I, mat* f, mat* F, mat* G, mat* grad_G);
void F_exp_calc(double h, mat* g, mat* I, mat* f, mat* F, mat* G, mat* grad_G);
void map_potential_partials(mat* C, mat* r, parameters inputs, mat* du_dr, mat* M);
mat hou_ode(mat x, mat t, parameters inputs);
void hill_solar_grav(mat* NA, mat* pos, mat* vel, double* n, mat* acc);
void md_tidal_torque(mat* pos, mat* vel, mat* w1, mat* w2, mat* NA, mat* AB, parameters inputs);
mat tilde_op(double x, double y, double z);
mat tilde_opv(mat A);
double factorial(double x);
void ic_read(initialization* ics);
vec kepler2cart(double* a_hyp, double* e_hyp, double* i_hyp, double* RAAN_hyp, double* om_hyp, double f0_hyp, double* G, double* Mplanet);
double kepler(double* n_hyp, double t, double* e_hyp, double* tau_hyp);
void grav_3BP(vec R_s, mat* NA, mat* pos, double* nu, double* G, double* Mplanet, mat* acc_3BP);
void solar_accel(vec R_s, mat* NA, mat* pos, double* nu, double* G, double* Msun, mat* acc_solar);

int main() {
    cout.precision(16);
    ic_read(&ics);//read input file
    int order=ics.order;
    int order_a=ics.order_a;
    int order_b=ics.order_b;
    //various matrix declarations 
    mat tk(order+1,order/2+2),IA(1,3),C(3,3),Cc(3,3),Cs(3,3),IB(1,3),sg_acc(3,1),acc_3BP(3,1),acc_solar(3,1),tt_1(3,1),tt_2(3,1),tt_orbit(3,1);
    mat IdA(3,3), IdB(3,3);
    IB.zeros();// There is a bug in Armadillo which results in IB having a value of NAN for its second entry if this call is not here, I've only seen this issue on Nexus, but it should be looked into at some point - Alex Davis
    cube TA,TB,TBp,TS,Tsun;
    field<cube> dT(3,3);
    sp_mat a(1,t_ind(order,order,order,order,order,order,order,order+1)+1);//max dimension of object with armadillo is 3, a nd b sets need to be 7 dimensional to t_ind func is used to map 7d into 2d matrix
    sp_mat b(1,t_ind(order,order,order,order,order,order,order,order+1)+1);
    string tet_fileA=ics.tfa;
    string vert_fileA=ics.vfa;
    string tet_fileB=ics.tfb;
    string vert_fileB=ics.vfb;


    double G=ics.G;
    double rhoA=ics.rhoA;
    double rhoB=ics.rhoB;
    int flyby_toggle=ics.flyby_toggle;
    int helio_toggle=ics.helio_toggle;
    int sg_toggle=ics.sg_toggle;
    int tt_toggle=ics.tt_toggle;
    double Mplanet=ics.Mplanet;
    double a_hyp=ics.a_hyp;
    double e_hyp=ics.e_hyp;
    double i_hyp=ics.i_hyp;
    double RAAN_hyp=ics.RAAN_hyp;
    double om_hyp=ics.om_hyp;
    double tau_hyp=ics.tau_hyp;
    double n_hyp = sqrt((G*Mplanet)/((pow(abs(a_hyp),3))));
    double Msolar=ics.Msolar;
    double a_helio=ics.a_helio;
    double e_helio=ics.e_helio;
    double i_helio=ics.i_helio;
    double RAAN_helio=ics.RAAN_helio;
    double om_helio=ics.om_helio;
    double tau_helio=ics.tau_helio;
    double n_helio = sqrt((G*Msolar)/((pow(abs(a_helio),3))));
    double sol_rad=ics.sol_rad;
    double au_def=ics.au_def;
    double love1=ics.love1;
    double love2=ics.love2;
    double refrad1=ics.refrad1;
    double refrad2=ics.refrad2;
    double eps1=ics.eps1;
    double eps2=ics.eps2;
    double Msun=ics.Msun;
    if (ics.Tgen){//generates inertia integrals if flag is set to 1
        cout<<"Generating Inertia Integrals"<<endl;
        //conditional to check what primary shape is set to
        if (ics.a_shape==0){//sphere
            ell_mass_params_met(order, 0, rhoA, ics.aA, ics.bA, ics.cA, &IA, &TA);
        }
        else if (ics.a_shape==1){//ellipsoid
            ell_mass_params_met(order, order_a, rhoA, ics.aA, ics.bA, ics.cA, &IA, &TA);
        }
        else if (ics.a_shape==2){//polyhedron
            poly_inertia_met(order_a, rhoA, tet_fileA, vert_fileA, &TA);
            poly_moi_met(rhoA, tet_fileA, vert_fileA, &IA);
        }
        else {
            cout<<"Bad Primary Shape Input"<<endl; 
        }
        //conditional to check what secondary shape is set to
        if (ics.b_shape==0){//sphere
            ell_mass_params_met(order, 0, rhoB, ics.aB, ics.bB, ics.cB, &IB, &TB);
        }
        else if (ics.b_shape==1){//ellipsoid
            ell_mass_params_met(order, order_b, rhoB, ics.aB, ics.bB, ics.cB, &IB, &TB);
        }
        else if (ics.b_shape==2){//polyhedron
            poly_inertia_met(order_b, rhoB, tet_fileB, vert_fileB, &TB);
            poly_moi_met(rhoB, tet_fileB, vert_fileB, &IB);
        }
        else {
            cout<<"Bad Secondary Shape Input"<<endl; 
        }
        // save inertia integrals to files armadillo binary .mat files
        TA.save("TDP_"+std::to_string(ics.order)+".mat");
        TB.save("TDS_"+std::to_string(ics.order)+".mat");
        IA.save("IDP.mat");
        IB.save("IDS.mat");
        
        //transform 3d inertia integral sets into 2d so they can be written as .csv
        mat TA_mat=TA.slice(0);
        mat TB_mat=TB.slice(0);
        for (int i=1; i<TA.n_slices; i++){
            TA_mat=join_vert(TA_mat,TA.slice(i));
            TB_mat=join_vert(TB_mat,TB.slice(i));
        }
        //save inertia integrals and moments to .csv files to be used by python postprocessing
        TA_mat.save("TDP_"+std::to_string(ics.order)+".csv",csv_ascii);
        TB_mat.save("TDS_"+std::to_string(ics.order)+".csv",csv_ascii);
        IA.save("IDP.csv",csv_ascii);
        IB.save("IDS.csv",csv_ascii);

    }
    TA.load("TDP_"+std::to_string(ics.order)+".mat");
    TB.load("TDS_"+std::to_string(ics.order)+".mat");
    IA.load("IDP.mat");
    IB.load("IDS.mat");
    TS.zeros(size(TA));
    Tsun.zeros(size(TA));
    TS(0,0,0)=Mplanet;
    Tsun(0,0,0)=Msolar;
    // calculate expansion coefficients
    tk_calc(order,&tk);
    a_calc(order,&a);
    b_calc(order,&b);
    // get masses and mass ratio
    double Mc=TA(0,0,0);
    double Ms=TB(0,0,0);
    double m=Mc*Ms/(Mc+Ms);
    double nu=Ms/(Ms+Mc);
    double mean_motion = pow(G*(Msun+Mc+Ms)/pow(sol_rad*au_def,3.),.5);
    mat x0, x_sol, t_sol, h_sol, sun_sol;//set up input and output matrices for integrators
    //set up times for integration
    double t0=ics.t0;
    double tf=ics.tf;
    x0=ics.x0;//unpack initial states from input structure
    // set pointers for system parameter pointer structure (parameters inputs above)
    inputs.G=&G;
    inputs.IA=&IA;
    inputs.IB=&IB;
    inputs.IdA=&IdA;
    inputs.IdB=&IdB;
    inputs.TA=&TA;
    inputs.TB=&TB;
    inputs.TBp=&TBp;
    inputs.TS=&TS;
    inputs.Tsun=&Tsun;
    inputs.a=&a;
    inputs.b=&b;
    inputs.dT=&dT;
    inputs.m=&m;
    inputs.nu=&nu;
    inputs.n=&order;
    inputs.tk=&tk;
    inputs.flyby_toggle=&flyby_toggle;
    inputs.helio_toggle=&helio_toggle;
    inputs.sg_toggle=&sg_toggle;
    inputs.tt_toggle=&tt_toggle;
    inputs.Mplanet=&Mplanet;
    inputs.a_hyp=&a_hyp;
    inputs.e_hyp=&e_hyp;
    inputs.i_hyp=&i_hyp;
    inputs.RAAN_hyp=&RAAN_hyp;
    inputs.om_hyp=&om_hyp;
    inputs.tau_hyp=&tau_hyp;
    inputs.n_hyp=&n_hyp;
    inputs.Msolar=&Msolar;
    inputs.a_helio=&a_helio;
    inputs.e_helio=&e_helio;
    inputs.i_helio=&i_helio;
    inputs.RAAN_helio=&RAAN_helio;
    inputs.om_helio=&om_helio;
    inputs.tau_helio=&tau_helio;
    inputs.n_helio=&n_helio;
    inputs.sol_rad=&sol_rad;
    inputs.au_def=&au_def;
    inputs.love1=&love1;
    inputs.love2=&love2;
    inputs.refrad1=&refrad1;
    inputs.refrad2=&refrad2;
    inputs.rhoA=&rhoA;
    inputs.rhoB=&rhoB;
    inputs.eps1=&eps1;
    inputs.eps2=&eps2;
    inputs.acc_3BP=&acc_3BP;
    inputs.acc_solar=&acc_solar;
    inputs.sg_acc=&sg_acc;
    inputs.tt_1=&tt_1;
    inputs.tt_2=&tt_2;
    inputs.tt_orbit=&tt_orbit;
    inputs.mean_motion=&mean_motion;
    //set up integration time matrix
    mat t(1,1);
    t<<t0<<endr;

    cout<<"Integrating"<<endl;
    if (ics.integ==1){
        cout<<"RK4"<<endl;//RK4 integrator
        rk4_stack(t0, tf, x0, ics.h, inputs,&t_sol, &x_sol, &h_sol, &sun_sol, hou_ode);
    }
    else if (ics.integ==2){
        cout<<"LGVI"<<endl;//LGVI integrator
        LGVI_integ(t0, tf, x0, ics.h, inputs, &t_sol, &x_sol);
    }
    else if (ics.integ==3){
        cout<<"RK 7(8)"<<endl;//RK7 (8) integrator
        rk87(t0, tf, x0, ics.tol, inputs,&t_sol, &x_sol, hou_ode);
    }
    else if (ics.integ==4){
        cout<<"ABM"<<endl;// A-B-M integrator
        ABM(t0, tf, x0, ics.h, inputs,&t_sol, &x_sol, &h_sol, &sun_sol, hou_ode);
    }
    // the below commented sections is the post-processing loop in cpp and can be uncommented for use if the user desires - this has not been rigorously tested as this file is designed for the wrapping described in the documentation
   // cout<<"Post-Processing"<<endl;
   // mat C_err(t_sol.n_rows,2);
   // double f,E0,H0;
   // for (uword t=0; t<t_sol.n_rows;t++){//1;t++){//
   //     f=t_sol(t,0);
   //     mat cc=reshape(x_sol(t,span(12,20)),3,3);
   //     cc=cc.t();
   //     mat c=reshape(x_sol(t,span(21,29)),3,3);
   //     c=c.t();
   //     mat cs=cc*c;
   //     mat r=cc*x_sol(t,span(0,2)).t();
   //     mat vel=cc*x_sol(t,span(3,5)).t();
   //     mat wc=x_sol(t,span(6,8)).t();
   //     mat ws=c.t()*x_sol(t,span(9,11)).t();
   //     double kt=norm(.5*m*vel.t()*vel,2);
   //     double kr1=norm(.5*wc.t()*diagmat(IA)*wc,2);
   //     double kr2=norm(.5*ws.t()*diagmat(IB)*ws,2);
   //     double H=norm(m*cross(r,vel)+cc*diagmat(IA)*wc+cs*diagmat(IB)*ws,2);
   //     mat e=cc.t()*(r/norm(r,2));
   //     double R=norm(r,2);
   //     inertia_rot(c, *(inputs.n), inputs.TB, inputs.TBp);
   //     double U=potential(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e.t(), R, inputs.TA, inputs.TBp);
//        cout<<U<<endl;
//        cout<<G<<endl;
//        e.raw_print();
//        cout<<R<<endl;
///        TA.raw_print();
//        TBp.raw_print();
//        cout<<R<<endl;
//        e.raw_print();
       // if (t>0){
       //     C_err(t,0)=(E0-(U+kr1+kr2+kt))/E0;
       //     C_err(t,1)=(H0-H)/H0;
       //     if(t==t_sol.n_rows-1){
//                (*(inputs.TBp)).raw_print();
//                c.raw_print();
       //     }
       // }
       // else{
       //     E0=U+kr1+kr2+kt;
       //     H0=H;
       //     C_err(t,0)=0.;
       //     C_err(t,1)=0.;
       // }
       
   // }
   // C_err.save("const_errs_out.csv",csv_ascii);
   // t_sol.save("t_out.csv",csv_ascii);
//    x_sol.save("x_out.csv",csv_ascii);
    return 0;
}

//generate tk expansion coefficients where m is mutual potential truncation order - output tk is matrix with rows of expansion order and columns of recursion steps
void tk_calc(int m, mat* t){
    for(int n=0; n<m+1; n++){//loop through expansion orders up to truncation order
        if (n%2){//if odd
            (*t)(n,0)=pow(-1.,(n-1.)/2.)*factorial((double)n)/(pow(2.,n-1.)
                   *pow(factorial((n-1.)/2.),2.)); 
        }
        else{//if even
            (*t)(n,0)=pow(-1.,n/2.)*factorial((double)n)/(pow(2.,n)
                    *pow(factorial(n/2.),2.));
        }
        //set recursion variables
        double k=fmod(n,2.);
        int i =1;
        while (k<=n){//recursion loop
            (*t)(n,i)=-(n-k)*(n+k+1.)*((*t)(n,i-1))/((k+2.)*(k+1.));
            k+=2.;
            i+=1;
        }
    }
    return;
}

//t ind maps the 7 indices of the a and b coefficient sets into a 2d matrix. a-g are i or j indices and dim is the k index as described in the Hou paper
int t_ind(int a, int b, int c, int d, int e, int f, int g, int dim){
    int ind;
    if (a+b+c+d+e+f+g>pow(dim,7)){
        ind=(dim-a)*pow(dim,6)+(dim-b)*pow(dim,5)+(dim-c)*pow(dim,4)+(dim-d)*pow(dim,3)+(dim-e)*pow(dim,2)+(dim-f)*dim+g;
    }
    else{
        ind=a*pow(dim,6)+b*pow(dim,5)+c*pow(dim,4)+d*pow(dim,3)+e*pow(dim,2)+f*dim+g;
    }
    return ind;
}

//calculates the a expansion coefficients based on the mutual potential truncation order n and a 2d matrix a
void a_calc(int n, sp_mat* a){
    (*a)(0,0)=1;//initialize a with values in Hou
    if (n>0){
        (*a)(0,t_ind(1,1,0,0,0,0,0,n+1))=1;
        (*a)(0,t_ind(1,0,1,0,0,0,0,n+1))=1;
        (*a)(0,t_ind(1,0,0,1,0,0,0,n+1))=1;
        (*a)(0,t_ind(1,0,0,0,1,0,0,n+1))=-1;
        (*a)(0,t_ind(1,0,0,0,0,1,0,n+1))=-1;
        (*a)(0,t_ind(1,0,0,0,0,0,1,n+1))=-1;
        if(n>1){
            for(int k=2; k<n+1; k++){// loop through k's and i's as described by Hou paper constraint equations
                for(int i1=0; i1<k+1; i1++){
                    for(int i2=0; i2<k+1-i1; i2++){
                        for(int i3=0; i3<k+1-i1-i2; i3++){
                            for(int i4=0; i4<k+1-i1-i2-i3; i4++){
                                for(int i5=0; i5<k+1-i1-i2-i3-i4; i5++){
                                    for(int i6=0; i6<k+1-i1-i2-i3-i4-i5; i6++){
                                        //conditionals used to check if indices are greater than 0 then performs recursive coefficient calculation
                                        if (i1>0){
                                           (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))+=
                                                   (*a)(0,t_ind(k-1,i1-1,i2,i3,i4,i5,i6,n+1));
                                        }
                                        if (i2>0){
                                            (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))+=
                                                    (*a)(0,t_ind(k-1,i1,i2-1,i3,i4,i5,i6,n+1));
                                        }
                                        if (i3>0){
                                            (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))+=
                                                    (*a)(0,t_ind(k-1,i1,i2,i3-1,i4,i5,i6,n+1));
                                            
                                        }
                                        if (i4>0){
                                            (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))-=
                                                    (*a)(0,t_ind(k-1,i1,i2,i3,i4-1,i5,i6,n+1));
                                        }
                                        if (i5>0){
                                           (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))-=
                                                   (*a)(0,t_ind(k-1,i1,i2,i3,i4,i5-1,i6,n+1)); 
                                        }
                                        if (i6>0){
                                            (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))-=
                                                    (*a)(0,t_ind(k-1,i1,i2,i3,i4,i5,i6-1,n+1));
                                        }}}}}}}}}}
    return;
}
//calculates the b expansion coefficients based on the mutual potential truncation order n and a 2d matrix b
void b_calc(int n, sp_mat* b){
    (*b)(0,0)=1;//initialize a with values in Hou
    if (n>1){
        (*b)(0,t_ind(2,2,0,0,0,0,0,n+1))=1;
        (*b)(0,t_ind(2,0,2,0,0,0,0,n+1))=1;
        (*b)(0,t_ind(2,0,0,2,0,0,0,n+1))=1;
        (*b)(0,t_ind(2,0,0,0,2,0,0,n+1))=1;
        (*b)(0,t_ind(2,0,0,0,0,2,0,n+1))=1;
        (*b)(0,t_ind(2,0,0,0,0,0,2,n+1))=1;
        (*b)(0,t_ind(2,1,0,0,1,0,0,n+1))=-2;
        (*b)(0,t_ind(2,0,1,0,0,1,0,n+1))=-2;
        (*b)(0,t_ind(2,0,0,1,0,0,1,n+1))=-2;
            for(int k=n; k>-1; k--){// loop through k's and i's as described by Hou paper constraint equations
                for(int j1=0; j1<n-k+1; j1++){
                    for(int j2=0; j2<n-k+1-j1; j2++){
                        for(int j3=0; j3<n-k+1-j1-j2; j3++){
                            for(int j4=0; j4<n-k+1-j1-j2-j3; j4++){
                                for(int j5=0; j5<n-k+1-j1-j2-j3-j4; j5++){
                                    for(int j6=0; j6<n-k+1-j1-j2-j3-j4-j5; j6++){
                                        if ((n-k)>2){
//                                          conditionals used to check if indices are greater than 0 then performs recursive coefficient calculation
                                            if (j1>0 && j4>0){
                                            (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=
                                                    -2*(*b)(0,t_ind(n-k-2,j1-1,j2,j3,j4-1,j5,j6,n+1));
                                            }
                                            if (j2>0 && j5>0){
                                                (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=
                                                    -2*(*b)(0,t_ind(n-k-2,j1,j2-1,j3,j4,j5-1,j6,n+1));
                                            }
                                            if (j3>0 && j6>0){
                                                (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=
                                                    -2*(*b)(0,t_ind(n-k-2,j1,j2,j3-1,j4,j5,j6-1,n+1));
                                            }
                                            if (j1>1){
                                                (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=
                                                    (*b)(0,t_ind(n-k-2,j1-2,j2,j3,j4,j5,j6,n+1));
                                            }
                                            if (j2>1){
                                               (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=
                                                   (*b)(0,t_ind(n-k-2,j1,j2-2,j3,j4,j5,j6,n+1)); 
                                            }
                                            if (j3>1){
                                                (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=
                                                    (*b)(0,t_ind(n-k-2,j1,j2,j3-2,j4,j5,j6,n+1));
                                            }
                                            if (j4>1){
                                                (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=
                                                    (*b)(0,t_ind(n-k-2,j1,j2,j3,j4-2,j5,j6,n+1));
                                            }
                                            if (j5>1){
                                               (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=
                                                   (*b)(0,t_ind(n-k-2,j1,j2,j3,j4,j5-2,j6,n+1)); 
                                            }
                                            if (j6>1){
                                                (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=
                                                    (*b)(0,t_ind(n-k-2,j1,j2,j3,j4,j5,j6-2,n+1));
                                            }
                                        }}}}}}}}}
   return;
}

//computes Q term for inertia integral calculation as described in Hou using summation indices i j and k
double Q_ijk(double i, double j, double k){
    return factorial(i)*factorial(j)*factorial(k)/factorial(i+j+k+3.);
}

// computes summation over a tetrahedron defined by 3 coordinates (x,y,z), fourth (x,y,z) is assumed to be at (0,0,0). Computation is based on l,m,n inertia integral order
// l,m,n must be int
// x,y,z's are vertices of tet, assuming fourth is at origin/barycenter
double tet_sums(double l, double m, double n, double x1, double x2, double x3, double y1, double y2, double y3, double z1, double z2, double z3){
    double sum_val=0.;
    for(double i1=0.; i1<(l+1.); i1++){//llops through indices based on Hou defined constraint
        for(double j1=0.; j1<(l-i1+1.); j1++){
            for(double i2=0.; i2<(m+1.); i2++){
                for(double j2=0.; j2<(m-i2+1.); j2++){
                    for(double i3=0.; i3<(n+1.);i3++){
                        for(double j3=0.; j3<(n-i3+1.); j3++){
                            sum_val+=(factorial(l)/(factorial(i1)*factorial(j1)
                                *factorial(l-i1-j1)))*(factorial(m)/(factorial(i2)
                                *factorial(j2)*factorial(m-i2-j2)))
                                *(factorial(n)/(factorial(i3)*factorial(j3)
                                *factorial(n-i3-j3)))
                                *pow(x1,i1)*pow(x2,j1)*pow(x3,(l-i1-j1))*pow(y1,i2)
                                *pow(y2,j2)*pow(y3,(m-i2-j2))*pow(z1,i3)*pow(z2,j3)
                                *pow(z3,(n-i3-j3))
                                *Q_ijk(i1+i2+i3,j1+j2+j3,l+m+n-i1-i2-i3-j1-j2-j3);
                        }}}}}}
    return sum_val;
}
// computes inertia integrals from tet and vert files defined in km, files are assumed to define polyhedron in their principle axis of inertia frame
// l,m,n must be int
// rho should be float of density
// tet_file must refer to a csv file with .csv included of tetrahedron defined by 3 vertex numbers
// vert_file must refer to a csv file with .csv included of vertex coords (x,y,z)
// here we assume vert has 4 columns where first column is index 
// q is truncation order n
void poly_inertia(int q, double rho, string tet_file, string vert_file, cube* T){
    mat tet,vert,x1,x2,x3,x;
    double Ta;
    (*T).zeros(q+1,q+1,q+1);
    tet.load(tet_file,csv_ascii);
    tet-=1;//accounts for indexing from 0
    vert.load(vert_file,csv_ascii);
    for(double l=0;l<(q+1);l++){//loops through inertia integral indices l,m,n up to q
        for(double m=0;m<(q+1-l);m++){
            for(double n=0;n<(q+1-l-m);n++){
                for(int a=0;a<tet.n_rows;a++){
                    x1=vert((int)tet(a,0),span(1,3));//get tetrahedron vertices in km!
                    x2=vert((int)tet(a,1),span(1,3));
                    x3=vert((int)tet(a,2),span(1,3));
                    x=join_vert(x1,x2);
                    x=join_vert(x,x3);
                    Ta=abs(det(x));
                    (*T)(l,m,n)+=rho*Ta*tet_sums(l,m,n,x1(0,0),x2(0,0),x3(0,0),x1(0,1),x2(0,1),x3(0,1),x1(0,2),x2(0,2),x3(0,2));//pass inertia integral indices and tet vertices in x,y,z coords
                }}}}
    return;
}
// computes inertia integrals from tet and vert files defined in meters, files are assumed to 
// define polyhedron in their principle axis of inertia frame
// l,m,n must be int
// rho should be float of density
// tet_file must refer to a csv file with .csv included of tetrahedron defined by 3 vertex numbers
// vert_file must refer to a csv file with .csv included of vertex coords (x,y,z)
// here we assume vert has 4 columns where first column is index 
// q is truncation order n
void poly_inertia_met(int q, double rho, string tet_file, string vert_file, cube* T){
    mat tet,vert,x1,x2,x3,x;
    double Ta;
    (*T).zeros(q+1,q+1,q+1);
    tet.load(tet_file,csv_ascii);
    tet-=1;//accounts for indexing from 0
    vert.load(vert_file,csv_ascii);
    vert=vert/1000.;// convert from meters into km form
    for(double l=0;l<(q+1);l++){//loops through inertia integral l,m,n indices up to q
        for(double m=0;m<(q+1-l);m++){
            for(double n=0;n<(q+1-l-m);n++){
                for(int a=0;a<tet.n_rows;a++){
                    x1=vert((int)tet(a,0),span(1,3));// get tetrahedron vertices now converted into km
                    x2=vert((int)tet(a,1),span(1,3));
                    x3=vert((int)tet(a,2),span(1,3));
                    x=join_vert(x1,x2);
                    x=join_vert(x,x3);
                    Ta=abs(det(x));
                    (*T)(l,m,n)+=rho*Ta*tet_sums(l,m,n,x1(0,0),x2(0,0),x3(0,0),x1(0,1),x2(0,1),x3(0,1),x1(0,2),x2(0,2),x3(0,2));//pass inertia integral indices and tet vertices in x,y,z coords
                }}}}
    return;
}
// inertia integral rotation function, takes in a standard rotation matrix, truncation order and principally aligned inertia integral set
// C is rotation matrix
// q is truncation order n
// T in inertia integral set to be rotated
void inertia_rot(mat C, int q, cube* T, cube* Tp){
    (*Tp).zeros(q+1,q+1,q+1);
    for(int l=0;l<(q+1);l++){//loop through inertia integral indices up to q
        for(int m=0;m<(q+1-l);m++){
            for(int n=0;n<(q+1-l-m);n++){
                for(double i1=0.; i1<(l+1.); i1++){//llop through i's and j's as defined in Hou summation constraint
                    for(double j1=0.; j1<(l-i1+1.); j1++){
                        for(double i2=0.; i2<(m+1.); i2++){
                            for(double j2=0.; j2<(m-i2+1.); j2++){
                                for(double i3=0.; i3<(n+1.);i3++){
                                    for(double j3=0.; j3<(n-i3+1.); j3++){
                                        if(((i1+i2+i3)<=q)&&((j1+j2+j3)<=q)&&((l+m+n-i1-i2-i3-j1-j2-j3)<=q)){// enforce i and j constraints
                                                (*Tp)(l,m,n)+=(factorial((double)l)/(factorial((double)i1)*factorial((double)j1)
                                                    *factorial((double)l-i1-j1)))
                                                    *(factorial((double)m)/(factorial((double)i2)*factorial((double)j2)
                                                    *factorial((double)(m-i2-j2))))
                                                    *(factorial((double)n)/(factorial((double)i3)*factorial((double)j3)
                                                    *factorial((double)(n-i3-j3))))
                                                    *pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))
                                                    *pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))
                                                    *pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3))
                                                    *(*T)(i1+i2+i3,j1+j2+j3,l+m+n-i1-i2-i3-j1-j2-j3);
                                        }}}}}}}}}}
    return;										
}

// polyhedron moment of inertia calculator takes in kg/km3 density, and km based tet and vert files for polyhedron assuming they are defined in principal frame
// this is not from Hou paper - comes from math paper deriving moments of inertia from point cloud
void poly_moi(double rho, string tet_file, string vert_file, mat* I){
    mat tet,vert;
    tet.load(tet_file,csv_ascii);
    tet-=1;//accounts for indexing from 0
    vert.load(vert_file,csv_ascii);
    double V;
    mat p1,p2,p3,p4,p;
    for(int a=0;a<tet.n_rows;a++){
        p1.zeros(1,3);//assuming all tet have vertex at origin
        p2=vert((int)tet(a,0),span(1,3));//get other 3 vertices
        p3=vert((int)tet(a,1),span(1,3));
        p4=vert((int)tet(a,2),span(1,3));
        p=join_vert(p2,p3);
        p=join_vert(p,p4);
        V=rho*abs(det(p))/6.;//determinant of vertices from lit search
        (*I)(0,0)+=V*(pow(p1(0,1),2)+p1(0,1)*p2(0,1)+pow(p2(0,1),2)+p1(0,1)*p3(0,1)+p2(0,1)*p3(0,1)+pow(p3(0,1),2)+
                pow(p1(0,2),2)+p1(0,2)*p2(0,2)+pow(p2(0,2),2)+p1(0,2)*p3(0,2)+p2(0,2)*p3(0,2)+pow(p3(0,2),2)+
		p1(0,1)*p4(0,1)+p2(0,1)*p4(0,1)+p3(0,1)*p4(0,1)+pow(p4(0,1),2)+
		p1(0,2)*p4(0,2)+p2(0,2)*p4(0,2)+p3(0,2)*p4(0,2)+pow(p4(0,2),2))/10.;//Ixx
        (*I)(0,1)+=V*(pow(p1(0,0),2)+p1(0,0)*p2(0,0)+pow(p2(0,0),2)+p1(0,0)*p3(0,0)+p2(0,0)*p3(0,0)+pow(p3(0,0),2)+
		pow(p1(0,2),2)+p1(0,2)*p2(0,2)+pow(p2(0,2),2)+p1(0,2)*p3(0,2)+p2(0,2)*p3(0,2)+pow(p3(0,2),2)+
		p1(0,0)*p4(0,0)+p2(0,0)*p4(0,0)+p3(0,0)*p4(0,0)+pow(p4(0,0),2)+
		p1(0,2)*p4(0,2)+p2(0,2)*p4(0,2)+p3(0,2)*p4(0,2)+pow(p4(0,2),2))/10.;//Iyy
	(*I)(0,2)+=V*(pow(p1(0,1),2)+p1(0,1)*p2(0,1)+pow(p2(0,1),2)+p1(0,1)*p3(0,1)+p2(0,1)*p3(0,1)+pow(p3(0,1),2)+
		pow(p1(0,0),2)+p1(0,0)*p2(0,0)+pow(p2(0,0),2)+p1(0,0)*p3(0,0)+p2(0,0)*p3(0,0)+pow(p3(0,0),2)+
		p1(0,1)*p4(0,1)+p2(0,1)*p4(0,1)+p3(0,1)*p4(0,1)+pow(p4(0,1),2)+
		p1(0,0)*p4(0,0)+p2(0,0)*p4(0,0)+p3(0,0)*p4(0,0)+pow(p4(0,0),2))/10.;//Izz
    }
    return;
}
// polyhedron moment of inertia calculator takes in kg/km3 density, and meter based tet and vert files for polyhedron assuming they are defined in principal frame
// // this is not from Hou paper - comes from math paper deriving moments of inertia from point cloud
void poly_moi_met(double rho, string tet_file, string vert_file, mat* I){
    mat tet,vert;
    tet.load(tet_file,csv_ascii);
    tet-=1;//accounts for indexing from 0
    vert.load(vert_file,csv_ascii);
    vert=vert/1000.;//convert from km to meters
    double V;
    mat p1,p2,p3,p4,p;
    for(int a=0;a<tet.n_rows;a++){
        p1.zeros(1,3);//assuming all tet have vertex at origin
        p2=vert((int)tet(a,0),span(1,3));//get other 3 vertices
        p3=vert((int)tet(a,1),span(1,3));
        p4=vert((int)tet(a,2),span(1,3));
        p=join_vert(p2,p3);
        p=join_vert(p,p4);
        V=rho*abs(det(p))/6.;
        (*I)(0,0)+=V*(pow(p1(0,1),2)+p1(0,1)*p2(0,1)+pow(p2(0,1),2)+p1(0,1)*p3(0,1)+p2(0,1)*p3(0,1)+pow(p3(0,1),2)+
                pow(p1(0,2),2)+p1(0,2)*p2(0,2)+pow(p2(0,2),2)+p1(0,2)*p3(0,2)+p2(0,2)*p3(0,2)+pow(p3(0,2),2)+
		p1(0,1)*p4(0,1)+p2(0,1)*p4(0,1)+p3(0,1)*p4(0,1)+pow(p4(0,1),2)+
		p1(0,2)*p4(0,2)+p2(0,2)*p4(0,2)+p3(0,2)*p4(0,2)+pow(p4(0,2),2))/10.;//Ixx
        (*I)(0,1)+=V*(pow(p1(0,0),2)+p1(0,0)*p2(0,0)+pow(p2(0,0),2)+p1(0,0)*p3(0,0)+p2(0,0)*p3(0,0)+pow(p3(0,0),2)+
		pow(p1(0,2),2)+p1(0,2)*p2(0,2)+pow(p2(0,2),2)+p1(0,2)*p3(0,2)+p2(0,2)*p3(0,2)+pow(p3(0,2),2)+
		p1(0,0)*p4(0,0)+p2(0,0)*p4(0,0)+p3(0,0)*p4(0,0)+pow(p4(0,0),2)+
		p1(0,2)*p4(0,2)+p2(0,2)*p4(0,2)+p3(0,2)*p4(0,2)+pow(p4(0,2),2))/10.;//Iyy
	(*I)(0,2)+=V*(pow(p1(0,1),2)+p1(0,1)*p2(0,1)+pow(p2(0,1),2)+p1(0,1)*p3(0,1)+p2(0,1)*p3(0,1)+pow(p3(0,1),2)+
		pow(p1(0,0),2)+p1(0,0)*p2(0,0)+pow(p2(0,0),2)+p1(0,0)*p3(0,0)+p2(0,0)*p3(0,0)+pow(p3(0,0),2)+
		p1(0,1)*p4(0,1)+p2(0,1)*p4(0,1)+p3(0,1)*p4(0,1)+pow(p4(0,1),2)+
		p1(0,0)*p4(0,0)+p2(0,0)*p4(0,0)+p3(0,0)*p4(0,0)+pow(p4(0,0),2))/10.;//Izz
    }
    return;
}

// Compute inertia integrals for ellipsoidal bodies using closed form 4th order calculation from Macmillan Theory of the Potential
// inputs: order - mutual potential truncation order
// order_body - inertia integral truncation order
// rho - kg/km3 density
// a,b,c are semi axes passed in in meters
// I pointer to inertia moments
// T pointer to inertia integrals
void ell_mass_params_met(double order, double order_body, double rho, double a, double b, double c, mat* I, cube* T){
    a=a/1000.;//convert to km
    b=b/1000.;
    c=c/1000.;
    double M=4.*rho*M_PI*a*b*c/3.;//calc mass based on ellipsoidal volume
    (*T).zeros(order+1,order+1,order+1);//set up inertia integrals and moments
    (*T)(0,0,0)=M;
    (*I)(0,0)=M*(pow(b,2)+pow(c,2))/5.;
    (*I)(0,1)=M*(pow(a,2)+pow(c,2))/5.;
    (*I)(0,2)=M*(pow(b,2)+pow(a,2))/5.;
 
    //conditionals to calculate inertia integrals based on closed form solution. Closed form shows ellipsoids have 0's for odd ordered inertia integrals
    if (order_body > 0){
        (*T)(2,0,0)=M*pow(a,2)/5.;
        (*T)(0,2,0)=M*pow(b,2)/5.;
        (*T)(0,0,2)=M*pow(c,2)/5.;
        if (order_body > 3){
            (*T)(4,0,0)=3.*M*pow(a,4)/35.;
            (*T)(0,4,0)=3.*M*pow(b,4)/35.;
            (*T)(0,0,4)=3.*M*pow(c,4)/35.;
            (*T)(2,2,0)=M*pow(a,2)*pow(b,2)/35.;
            (*T)(0,2,2)=M*pow(c,2)*pow(b,2)/35.;
            (*T)(2,0,2)=M*pow(a,2)*pow(c,2)/35.;
        }
    }
    return;
}


// u_tilde takes the expansion order (not truncation order) n, the row of matrix tk corresponding to n (input as t), the a and b coefficient sets, 
// the relative position unit vector e, primary inertia integral set TA and rotated secondary inertia integral set TBp - units of TA and TBp should be km, kg
// if you see a bug from indexing here, check the order being input into all functions
// e must be a row vector!!!!
// dim is the mutual potential truncation order used to size the a and b coefficients
double u_tilde(int dim,int n, mat* t, sp_mat* a, sp_mat* b, mat e, cube* TA, cube* TBp){
    mat u((*t).n_cols,1);
    u.zeros();// initialize u tilde as 0
    for(int k=n; k>-1; k-=2){// loop down from n by 2s
        for(int i1=0; i1<k+1; i1++){//lopp through i's and j's based on Hou constraint
            for(int i2=0; i2<k+1-i1; i2++){
                for(int i3=0; i3<k+1-i1-i2; i3++){
                    for(int i4=0; i4<k+1-i1-i2-i3; i4++){
                        for(int i5=0; i5<k+1-i1-i2-i3-i4; i5++){
                            for(int j1=0; j1<n-k+1; j1++){
                                for(int j2=0; j2<n-k+1-j1; j2++){
                                    for(int j3=0; j3<n-k+1-j1-j2; j3++){
                                        for(int j4=0; j4<n-k+1-j1-j2-j3; j4++){
                                            for(int j5=0; j5<n-k+1-j1-j2-j3-j4; j5++){
                                                int i6=k-i1-i2-i3-i4-i5;
                                                int j6=n-k-j1-j2-j3-j4-j5;
                                                u(k/2,0)+=(*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,dim+1))
                                                    *(*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,dim+1))
                                                    *pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))
                                                    *pow(e(0,2),(i3+i6))
                                                    *(*TA)(i1+j1,i2+j2,i3+j3)
                                                    *(*TBp)(i4+j4,i5+j5,i6+j6);
                                                }}}}}}}}}}
        u(k/2,0)=u(k/2,0)*(*t)(n,k/2);//handle summation over tk
    }
    return accu(u);
}

// Calculates the partial of u tilde with respect to element dx (integer index) of the full position vector
// inputs: n - mutual potential expansion order (not truncation order)
// t - row n of tk matrix (it is the tk coefficients corresponding to order n
// a,b - a and be coefficient arrays
// e - unit vector of relative position in same frame as TA (must be row oriented)
// R - magnitude of rel pos in km
// dx - element of full rel position (0,1,2) the partial is taken with respect to
// dim - mutual potential truncation order used to size the a and b coefficients
// outputs: du_dx_tilde - km kg s unit partial of u tilde with respect to x
double du_dx_tilde(int dim,int n, mat* t, sp_mat* a, sp_mat* b, mat e, double R, int dx, cube* TA, cube* TBp){
    mat du((*t).n_cols,1);
    double ce, de_dx0,de_dx1,de_dx2;
    du.zeros();
    de_dx0=de_dx(e,R,0,dx);//get partials of e elements with respect to dx
    de_dx1=de_dx(e,R,1,dx);
    de_dx2=de_dx(e,R,2,dx);
    for(int k=n; k>-1; k-=2){//loop down by 2's from n
        for(int i1=0; i1<k+1; i1++){//loop through i's and j's based on Hou paper constraint
            for(int i2=0; i2<k+1-i1; i2++){
                for(int i3=0; i3<k+1-i1-i2; i3++){
                    for(int i4=0; i4<k+1-i1-i2-i3; i4++){
                        for(int i5=0; i5<k+1-i1-i2-i3-i4; i5++){
                            for(int j1=0; j1<n-k+1; j1++){
                                for(int j2=0; j2<n-k+1-j1; j2++){
                                    for(int j3=0; j3<n-k+1-j1-j2; j3++){
                                        for(int j4=0; j4<n-k+1-j1-j2-j3; j4++){
                                            for(int j5=0; j5<n-k+1-j1-j2-j3-j4; j5++){
                                                int i6=k-i1-i2-i3-i4-i5;
                                                int j6=n-k-j1-j2-j3-j4-j5;
                                                // conditionals to catch partials that go do 0 in overall partial of u tilde where ce accounts for the e portion of the u tilde equation
                                                if(i1+i4==0){
                                                    if(i2+i5==0){
                                                        if (i3+i6==0){
                                                        ce=0.;
                                                        }
                                                        else{
                                                            ce=(i3+i6)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))
                                                                *pow(e(0,2),(i3+i6-1.))*de_dx2;
                                                        }
                                                    }
                                                    else{
                                                        if (i3+i6==0){
                                                            ce=(i2+i5)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5-1.))
                                                                    *pow(e(0,2),(i3+i6))*de_dx1;
                                                        }
                                                        else{
                                                            ce=(i3+i6)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))
                                                                    *pow(e(0,2),(i3+i6-1.))*de_dx2
                                                                    +(i2+i5)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5-1.))
                                                                    *pow(e(0,2),(i3+i6))*de_dx1;
                                                        }}}
                                                else{
                                                    if(i2+i5==0){
                                                        if(i3+i6==0){
                                                            ce=(i1+i4)*pow(e(0,0),(i1+i4-1.))*pow(e(0,1),(i2+i5))
                                                                    *pow(e(0,2),(i3+i6))*de_dx0;
                                                        }
                                                        else{
                                                            ce=(i3+i6)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))
                                                                    *pow(e(0,2),(i3+i6-1.))*de_dx2
                                                                    +(i1+i4)*pow(e(0,0),(i1+i4-1.))*pow(e(0,1),(i2+i5))
                                                                    *pow(e(0,2),(i3+i6))*de_dx0;
                                                        }
                                                    }
                                                    else{
                                                        if(i3+i6==0){
                                                            ce=(i1+i4)*pow(e(0,0),(i1+i4-1.))*pow(e(0,1),(i2+i5))
                                                                    *pow(e(0,2),(i3+i6))*de_dx0
                                                                    +(i2+i5)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5-1.))
                                                                    *pow(e(0,2),(i3+i6))*de_dx1;
                                                        }
                                                        else{
                                                            ce=(i1+i4)*pow(e(0,0),(i1+i4-1.))*pow(e(0,1),(i2+i5))
                                                                    *pow(e(0,2),(i3+i6))*de_dx0
                                                                    +(i2+i5)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5-1.))
                                                                    *pow(e(0,2),(i3+i6))*de_dx1
                                                                    +(i3+i6)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))
                                                                    *pow(e(0,2),(i3+i6-1.))*de_dx2;
                                                        }}}
                                                du(k/2,0)+=(*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,dim+1))
                                                        *(*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,dim+1))
                                                        *ce
                                                        *(*TA)(i1+j1,i2+j2,i3+j3)
                                                        *(*TBp)(i4+j4,i5+j5,i6+j6);
                                                    }}}}}}}}}}
        du(k/2,0)=du(k/2,0)*(*t)(n,k/2);//handle tk part of summation in the Hou paper
    }
    return accu(du);
}

// partial of relative position unit vector with respect to full position
// pass in relative poition unit vector e, relative position magnitude R in km, element of e (de) that partial with respect to element of full position vector (dx)  
//de is index(int) corresponding to de subscript
//dx is index(int) corresponding to dx subscript
double de_dx(mat e, double R, int de, int dx){
    	mat x=R*e;//make full position vector
        double val;
	if (de==dx){// check which form of partial of e elements should be used
		int ind[]={0,1,2};
		for (int i=dx;i<2;i++){
                    ind[i]=ind[i+1];
                }
		val=(pow(x(0,ind[0]),2)+pow(x(0,ind[1]),2))/pow(R,3);
        }
	else{
		val=-x(0,de)*x(0,dx)/pow(R,3);
        }
	return val;
}

// Partial of rotated inertia integral set with respect to rotation matrix element C(i,j)
//inputs: ij - indices of C(i,j) identifying rotation matrix element
//C - rotation matrix
//q - truncation order of inertia integral expansion
//T - inertia integral set being rotated
//ij should be a list of 2 values, the first being i the second being j
void dT_dc(int i, int j, mat C, int q, cube* TA, cube* dT){
    double c;
    (*dT).zeros(q+1,q+1,q+1);
    for(int l=0;l<(q+1);l++){// loop through inertia integral indices
        for(int m=0;m<(q+1-l);m++){
            for(int n=0;n<(q+1-l-m);n++){
                for(double i1=0.; i1<(l+1.); i1++){// loop through i's and j's based on constraints in Hou paper
                    for(double j1=0.; j1<(l-i1+1.); j1++){
                        for(double i2=0.; i2<(m+1.); i2++){
                            for(double j2=0.; j2<(m-i2+1.); j2++){
                                for(double i3=0.; i3<(n+1.);i3++){
                                    for(double j3=0.; j3<(n-i3+1.); j3++){
                                        if(((i1+i2+i3)<=q)&&((j1+j2+j3)<=q)&&((l+m+n-i1-i2-i3-j1-j2-j3)<=q)){//ensure constraints are enforced
                                            c=0.;// conditionals used to catch coeefficient which go to 0 in partial of dT equation, c accounts for rotation matrix portion of inertia integral rotation equation
                                            if (i==0){
                                                if (j==0 && i1>0){
                                                    c=i1*pow(C(0,0),(i1-1.))*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))
                                                            *pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))
                                                            *pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                }
                                                else if(j==1 && j1>0){
                                                    c=j1*pow(C(0,0),i1)*pow(C(0,1),(j1-1.))*pow(C(0,2),(l-i1-j1)
                                                            )*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2)
                                                            )*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                }
                                                else if(j==2 && (l-j1-i1)>0){
                                                    c=(l-i1-j1)*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1-1.)
                                                            )*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2)
                                                            )*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                }
                                            }
                                            else if(i==1){
                                                if(j==0 && i2>0){
                                                    c=i2*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1)
                                                            )*pow(C(1,0),(i2-1.))*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2)
                                                            )*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                }
                                                else if(j==1 && j2>0){
                                                    c=j2*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1)\
                                                            )*pow(C(1,0),i2)*pow(C(1,1),(j2-1.))*pow(C(1,2),(m-i2-j2)
                                                            )*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                }
                                                else if(j==2 && (m-i2-j2)>0){
                                                    c=(m-i2-j2)*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1)
                                                            )*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2-1.)
                                                            )*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                }
                                            }
                                            else{
                                                if(j==0 && i3>0){
                                                    c=i3*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1)
                                                            )*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2)
                                                            )*pow(C(2,0),(i3-1.))*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                }
                                                else if(j==1 && j3>0){
                                                    c=j3*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1)
                                                            )*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2)
                                                            )*pow(C(2,0),i3)*pow(C(2,1),(j3-1.))*pow(C(2,2),(n-i3-j3));
                                                }
                                                else if(j==2 && (n-i3-j3)){
                                                    c=(n-i3-j3)*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1)
                                                            )*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))
                                                            *pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3-1.));
                                                }
                                            }
                                            (*dT)(l,m,n)+=(factorial((double)l)/(factorial((double)i1)*factorial((double)j1)
                                                    *factorial((double)l-i1-j1)))
                                                    *(factorial((double)m)/(factorial((double)i2)*factorial((double)j2)
                                                    *factorial((double)m-i2-j2)))\
                                                    *(factorial((double)n)/(factorial((double)i3)*factorial((double)j3)
                                                    *factorial(n-i3-j3)))
                                                    *c
                                                    *(*TA)(i1+i2+i3,j1+j2+j3,l+m+n-i1-i2-i3-j1-j2-j3);
                                        }}}}}}}}}}
    return;
}

// Calculates the partial of u tilde with respect to element C(i,j) of the rotation matrix
// inputs: n - mutual potential expansion order (not truncation order)
// t - row n of tk matrix (it is the tk coefficients corresponding to order n
// a,b - a and be coefficient arrays
// e - unit vector of relative position in same frame as TA (must be row oriented)
// TA - primary set of inertia integrals with units of km and kg
// TBp - rotated secondary inertia integrals with units of km and kg
// dT - partial of rotated secondary inertia integral set with respect to some C(i,j)
// dim - mutual potential truncation order used to size the a and b coefficients
// outputs: du_dx_tilde - km kg s unit partial of u tilde with respect to x
double du_dc_tilde(int dim,int n, mat* t, sp_mat* a, sp_mat* b, mat e, cube* TA, cube* dT){
    mat du((*t).n_cols,1);
    du.zeros();
    for(int k=n; k>-1; k-=2){//count down by 2's from n 
        for(int i1=0; i1<k+1; i1++){// loop over i's and j's based on constraint equation in Hou
            for(int i2=0; i2<k+1-i1; i2++){
                for(int i3=0; i3<k+1-i1-i2; i3++){
                    for(int i4=0; i4<k+1-i1-i2-i3; i4++){
                        for(int i5=0; i5<k+1-i1-i2-i3-i4; i5++){
                            for(int j1=0; j1<n-k+1; j1++){
                                for(int j2=0; j2<n-k+1-j1; j2++){
                                    for(int j3=0; j3<n-k+1-j1-j2; j3++){
                                        for(int j4=0; j4<n-k+1-j1-j2-j3; j4++){
                                            for(int j5=0; j5<n-k+1-j1-j2-j3-j4; j5++){
                                                int i6=k-i1-i2-i3-i4-i5;
                                                int j6=n-k-j1-j2-j3-j4-j5;
                                                du(k/2,0)+=(*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,dim+1))
                                                        *(*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,dim+1))
                                                        *pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))
                                                        *pow(e(0,2),(i3+i6))
                                                        *(*TA)(i1+j1,i2+j2,i3+j3)
                                                        *(*dT)(i4+j4,i5+j5,i6+j6);
                                                    }}}}}}}}}}
        du(k/2,0)=du(k/2,0)*(*t)(n,k/2);//handle tk portion of summation
    }
    return accu(du);
}

// Computes partial of potential with respect to element of full relative position dx
//inputs: G - gravity constant in units of km kg
//m - mutual potential truncation order
//t - full set of tk coefficients
//a,b - a and b expansion coefficients
//e - unit vector of relative position
//R - magnitude of relative position in km
//TA - primary inertia integrals
//TBp - secondary inertia integrals rotated into A frame
// returns partial of energy potential (negative of force and value in Hou paper)
double du_x(double G, int m, mat* t, sp_mat* a, sp_mat* b, mat e, double R, int dx, cube* TA, cube* TBp){
    double du=0.;
    mat x;
    x=e*R;//get full pos vec
    for (int n=0;n<m+1;n++){// loop frok N=0 to truncation order
        du+=(-((n+1.)*(x(0,dx)))/pow(R,(n+3.)))*u_tilde(m,n, t, a, b, e, TA, TBp)
                +(1./pow(R,(n+1.)))*du_dx_tilde(m,n, t, a, b, e, R, dx, TA, TBp);
    }
    return -du*G;
}

// Computes partial of potential with respect to element of rotation matrix C which maps from B to A
//inputs: G - gravity constant in units of km kg
//m - mutual potential truncation order
//t - full set of tk coefficients
//a,b - a and b expansion coefficients
//e - unit vector of relative position
//R - magnitude of relative position in km
//TA - primary inertia integrals
//TBp - secondary inertia integrals rotated into A frame
//dT - partial of secondary inertia integrals rotated into A frame with respect to rotation matrix element C(i,j) where C maps from B to A
// returns partial of energy potential (negative of equation in Hou paper)
double du_c(double G, int m, mat* t, sp_mat* a, sp_mat* b, mat e, double R, cube* TA, cube* dT){
    double du=0.;
    for (int n=0;n<m+1;n++){// loop frok N=0 to truncation order
        du+=(1./pow(R,(n+1.)))*du_dc_tilde(m,n,t,a,b,e,TA,dT);
    }
    return -du*G;
}

// Computes Energy potential
//inputs: G - gravity constant in units of km kg
//m - mutual potential truncation order
//t - full set of tk coefficients
//a,b - a and b expansion coefficients
//e - unit vector of relative position
//R - magnitude of relative position in km
//TA - primary inertia integrals
//TBp - secondary inertia integrals rotated into A frame
// returns partial of energy potential (negative of force)
double potential(double G, int m, mat* t, sp_mat* a, sp_mat* b, mat e, double R, cube* TA, cube* TBp){
//    cout<<(*t)<<endl;
    double u=0.;
    for (int n=0;n<m+1;n++){// loop frok N=0 to truncation order
        u+=(1./pow(R,(n+1.)))*u_tilde(m,n,t,a,b,e,TA,TBp);
//        cout<<m<<","<<n<<","<<(1./pow(R,(n+1.)))<<","<<u_tilde(m,n,t,a,b,e,TA,TBp)<< endl;
    }
    return -u*G;
}

//This is a standard implementation of the RK 7(8) Dormand Prince Integrator, the derivation and analysis of which
//can be found throughout the literature. To summarize the integrator is an adaptive integrator using the
//tolerance of the states to control the step size (comparisons between order 7 and 8). The output
//is binary files of the states and time written for each evaluation of the dynamics. The dynamics used by the integrator 
//can be changed using the ode function
//
//inputs: t0 - initial time (units of input states)
//tf - final time (units of the input states)
//x0 - row vector of the the states, the ode function must be set up to take in x0 in the form input to this function
//rel_tol - fractional tolerance between the states used for the adaptive step control
//inputs - structure of parameters which may be needed by the ode function.
//tout - pointer to a matrix which will be used to house the output times
//xout - point to a matrix which will be used to house the output states
//ode - function which takes as inputs the states as x0, the time as a scalar and the inputs structure
//outputs: the integrator will save two binary files in subfolders output_t and output_x (be careful these are deleted with each run).
//the binary files are single line wrappings of the tout and xout matrix, which is to say the each row of xout is entered in line in the output binary
//the same occurs for the output time
void rk87(double t0, double tf, mat x0, double rel_tol, parameters inputs, mat* tout, mat* xout, std::function<mat(mat,mat, parameters)> ode){
    //system commands which remove old integrated files and create new output directories
    system("if [ -d output_t ]; then rm -rf output_t; fi; mkdir output_t");
    system("if [ -d output_x ]; then rm -rf output_x; fi; mkdir output_x");
    ofstream tfile("output_t/t_out.bin", std::ios::binary);
    ofstream xfile("output_x/x_out.bin", std::ios::binary);
    mat c_i, a_i_j, b_8, b_7,sol1,sol2;
    double tau,err;
    //set up rk 7 and rk 8 dormand prince integrator coefficient matrices
    c_i << 1./18. << endr << 1./12. << endr << 1./8. << endr << 5./16. << endr
            << 3./8. << endr << 59./400. << endr << 93./200. << endr << 5490023248./9719169821. << endr
            << 13./20. << endr << 1201146811./1299019798. << endr << 1. << endr << 1. << endr;
    
    a_i_j << 1./18.<< 1./48.<<1./32. <<5./16. << 3./80.<<29443841./614563906. <<16016141./946692911. <<39632708./573591083. <<246121993./1340847787. <<-1028468189./846180014. <<185892177./718116043. <<403863854./491063109. << endr
            <<0. <<1./16. <<0. << 0.<< 0.<< 0.<< 0.<< 0.<< 0.<< 0.<< 0.<< 0.<< endr
            <<0. <<0. <<3./32. <<-75./64. <<0. <<0. <<0. << 0.<< 0.<< 0.<< 0.<< 0.<< endr
            <<0. <<0. << 0.<<75./64. <<3./16. <<77736538./692538347. <<61564180./158732637. <<-433636366./683701615. <<-37695042795./15268766246. << 8478235783./508512852.<< -3185094517./667107341.<< -5068492393./434740067.<< endr
            <<0. <<0. << 0.<<0. <<3./20. << -28693883./1125000000.<< 22789713./633445777.<< -421739975./2616292301.<< -309121744./1061227803.<< 1311729495./1432422823.<< -477755414./1098053517.<< -411421997./543043805.<< endr
            <<0. <<0. << 0.<<0. <<0. << 23124283./1800000000.<< 545815736./2771057229.<< 100302831./723423059.<< -12992083./490766935.<< -10304129995./1701304382.<< -703635378./230739211.<< 652783627./914296604.<< endr
            <<0. <<0. << 0.<<0. <<0. <<0. << -180193667./1043307555.<< 790204164./839813087.<< 6005943493./2108947869.<< -48777925059./3047939560.<< 5731566787./1027545527.<< 11173962825./925320556.<< endr
            <<0. <<0. << 0.<<0. <<0. <<0. <<0. << 800635310./3783071287.<< 393006217./1396673457.<< 15336726248./1032824649.<< 5232866602./850066563.<< -13158990841./6184727034.<< endr
            <<0. <<0. << 0.<<0. <<0. <<0. <<0. <<0. << 123872331./1001029789.<< -45442868181./3398467696.<< -4093664535./808688257.<< 3936647629./1978049680.<< endr
            <<0. << 0.<< 0.<<0. <<0. <<0. <<0. <<0. <<0. << 3065993473./597172653.<< 3962137247./1805957418.<< -160528059./685178525.<< endr
            <<0. <<0. << 0.<<0. <<0. <<0. <<0. <<0. <<0. <<0. << 65686358./487910083.<< 248638103./1413531060.<< endr
            <<0. << 0.<< 0.<<0. <<0. <<0. <<0. <<0. <<0. <<0. <<0. <<0. << endr
            <<0. << 0.<<0. <<0. <<0. <<0. <<0. <<0. <<0. <<0. <<0. <<0. << endr;

    b_8 << 14005451./335480064.<< endr << 0.<< endr << 0.<< endr << 0.<< endr << 0.<< endr <<-59238493./1068277825. << endr <<181606767./758867731. << endr << 561292985./797845732. <<endr<< -1041891430./1371343529.<< endr
            <<760417239./1151165299.<< endr << 118820643./751138087.<< endr <<-528747749./2220607170. << endr << 1./4.<<endr;
    
    b_7 << 13451932./455176623.<< endr << 0.<< endr << 0.<< endr << 0.<< endr << 0.<< endr <<-808719846./976000145. << endr <<  1757004468./5645159321.<< endr << 656045339./265891186.<<endr<< -3867574721./1518517206.<< endr
            <<465885868./322736535.<< endr << 53011238./667516719.<< endr <<2./45.<< endr << 0.<<endr;
    
    //set integration time step limits based on runtime
    double powv=1./8.;
    double hmax=(tf-t0)/2.5;
    double h=(tf-t0)/50.;
    //initial time step is forced to be a small value to error
    if (h>.1){
        h=.1;
    }
    if (h>hmax){
        h=hmax;
    }
    //initialize integrator parameters
    mat t(1,1);
    t<<t0<<endr;
    double eps=5.e-16;//machine precision
    double hmin=16.*eps*abs(t(0,0));//minimal value to avoid stepping below machine precision
    mat x=x0;
    mat f;
    //matrices are initialized to a limitted size. As the integrator reaches this limited size (1000 here)
    //it will write out the contents to the output binary's then zero out the matrix and begin refilling it
    //this is done to avoid large sets of data being manipulated in memory for long integrations
    f.zeros(x.n_cols,13);
    (*xout).resize(1000.,x.n_cols);//#(x.n_rows,x.n_cols);
    (*xout).zeros();
    (*xout).row(0)=x;
    (*tout).resize(1000.,1);
    (*tout).zeros();
    (*tout).row(0)=t;
    tau=rel_tol*norm(x,"inf");
    int store_count=0;
    int files=0;
    //begin integration
    while(t(0,0)<tf){
        if(t(0,0)+h>tf){//make sure integration goes exactly to end time
            h=tf-t(0,0);
        }
        f.col(0)=ode(x,t,inputs);
        // RK 7(8) dormand prince loop
        for(int j=1;j<13;j++){
            f.col(j)=ode(x+h*trans(f*a_i_j.col(j-1)),t+c_i(j-1,0)*h,inputs);
        }
        //set up comparison of order 7 and order 8 results
        sol1=x+h*trans(f*b_8);
        sol2=x+h*trans(f*b_7);
        err=abs(norm(sol1-sol2,"inf"));
        tau=rel_tol*norm(x,"inf");
        //adaptive step loop
        if (err<= tau){//if error is within the user specified record integration step
           store_count++;
           t=t+h;
           x=sol2;
           if (store_count<1000){//this conditional us used to write out to binaries when the output matrices are filled
               (*tout).row(store_count)=t;
               (*xout).row(store_count)=x;
           }
           else{//this is the print out to the binary files
               for (int indr=0;indr<(*xout).n_rows;indr++){
                   tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
                   for (int indc=0;indc<(*xout).n_cols;indc++){
                       xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
                   }
               }
               //now that data is printed to binary files we rezero storage matrix
               (*tout).zeros();
               (*xout).zeros();
               (*tout).row(0)=t;
               (*xout).row(0)=x;
               store_count=0;
               cout<<"Saving"<<endl;
               cout<<files<<endl;
               files++;
           }
        }
        if (err==0.){//check to see if the integrator is "perfect" then change adaptive time step so that integrator can run fast
            err=10.*eps;
        }
        h=fmin(hmax,.9*h*pow(tau/err,powv));
        if (abs(h)<=eps){//error check for when step size is hitting machine precision
            cout << "too small step" << endl;
        }
    }
    //now that integration loop is done write out any other output data to binary files
    if (store_count==0){
        (*tout)=t;
        (*xout)=x;
        for (int indr=0;indr<(*xout).n_rows;indr++){
            tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
            for (int indc=0;indc<(*xout).n_cols;indc++){
                xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
            }
        }
    }
    else{
        (*xout)=(*xout)(span(0,store_count),span(0,29));
        (*tout)=(*tout)(span(0,store_count),0);
        for (int indr=0;indr<(*xout).n_rows;indr++){
            tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
            for (int indc=0;indc<(*xout).n_cols;indc++){
                xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
            }
        }
    }
    //close binary files
    xfile.close();
    tfile.close();
    return;
}

//This is a standard implementation of the fixed step RK 4 Integrator, the derivation and analysis of which can be found throughout the literature.The output
//is binary files of the states and time written for each evaluation of the dynamics. The dynamics used by the integrator can be changed using the ode function
//
//inputs: t0 - initial time (units of input states)
//tf - final time (units of the input states)
//x0 - row vector of the the states, the ode function must be set up to take in x0 in the form input to this function
//h - integration time step (units of input states)
//inputs - structure of parameters which may be needed by the ode function.
//tout - pointer to a matrix which will be used to house the output times
//xout - point to a matrix which will be used to house the output states
//ode - function which takes as inputs the states as x0, the time as a scalar and the inputs structure
//outputs: the integrator will save two binary files in subfolders output_t and output_x (be careful these are deleted with each run).
//the binary files are single line wrappings of the tout and xout matrix, which is to say the each row of xout is entered in line in the output binary
//the same occurs for the output time
//if the flyby or heliocentric toggles are turned on, the integrator also saves binary files for the perturbing body relative to the binary system barycenter.
//hout and sunout and pointers to a matrix to house the states of the perturber (3rd body or the sun)
void rk4_stack(double t0, double tf, mat x0, double h, parameters inputs, mat* tout, mat* xout, mat* hout, mat*sunout, std::function<mat(mat,mat, parameters)> ode){
    //system commands which remove old integrated files and create new output directories
    system("if [ -d output_t ]; then rm -rf output_t; fi; mkdir output_t");
    system("if [ -d output_x ]; then rm -rf output_x; fi; mkdir output_x");
    system("if [ -d output_h ]; then rm -rf output_h; fi; mkdir output_h");
    system("if [ -d output_sun ]; then rm -rf output_sun; fi; mkdir output_sun");
    ofstream tfile("output_t/t_out.bin", std::ios::binary);
    ofstream xfile("output_x/x_out.bin", std::ios::binary);
    ofstream hfile("output_h/h_out.bin", std::ios::binary);
    ofstream sunfile("output_sun/sun_out.bin", std::ios::binary);

    //set up rk4 variables and k's for rk computation
    mat k1,k2,k3,k4,y0,y,next_y;
    mat t(1,1);
    t<<t0<<endr;
    y0=x0;
    //matrices are initialized to a limitted size. As the integrator reaches this limited size (the number of integration steps to reach one day in seconds)
    //it will write out the contents to the output binary's then zero out the matrix and begin refilling it
    //this is done to avoid large sets of data being manipulated in memory for long integrations
    (*xout).resize(int(3600*24/h),y0.n_cols);
    (*xout).row(0)=y0;
    (*tout).resize(int(3600*24/h),1);
    (*tout).row(0)=t(0,0);

    double f0_hyp = kepler(inputs.n_hyp, t0, inputs.e_hyp, inputs.tau_hyp);
    vec X_s = kepler2cart(inputs.a_hyp, inputs.e_hyp, inputs.i_hyp, inputs.RAAN_hyp, inputs.om_hyp, f0_hyp, inputs.G, inputs.Mplanet);
    (*hout).resize(int(3600*24/h),X_s.n_rows);
    (*hout).row(0)=X_s.t();

    double f0_helio = kepler(inputs.n_helio, t0, inputs.e_helio, inputs.tau_helio);
    vec X_helio = kepler2cart(inputs.a_helio, inputs.e_helio, inputs.i_helio, inputs.RAAN_helio, inputs.om_helio, f0_helio, inputs.G, inputs.Msolar);
    (*sunout).resize(int(3600*24/h),X_helio.n_rows);
    (*sunout).row(0)=X_helio.t();

    int count=0;
    int files=1;
    mat t_temp,y_temp;
    //begin integration
    while(t(0,0)<tf){
        t_temp=(*tout)(count,0);
        y_temp=(*xout).row(count);
        //rk4 steps
        k1=ode(y_temp,t_temp,inputs);
        k2=ode(y_temp+(h/2.)*k1.t(),t_temp+(h/2.),inputs);
        k3=ode(y_temp+(h/2.)*k2.t(),t_temp+(h/2.),inputs);
        k4=ode(y_temp+h*k3.t(),t_temp+h,inputs);
        //rk calculation
        next_y=y_temp+(h/6.)*(k1.t()+2.*k2.t()+2.*k3.t()+k4.t());
        if (count==(int(86400/h)-1)){//this is the print out to the binary files
            for (int indr=0;indr<(*xout).n_rows;indr++){
                tfile.write((char*) &((*tout)(indr,0)), sizeof(double));               
                for (int indc=0;indc<(*xout).n_cols;indc++){
                    xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
                }
                if(*inputs.flyby_toggle==1){
                    for (int indc=0;indc<(*hout).n_cols;indc++){
                        hfile.write((char*) &((*hout)(indr,indc)), sizeof(double));
                    }
                }   
                if(*inputs.helio_toggle==1){
                    for (int indc=0;indc<(*sunout).n_cols;indc++){
                        sunfile.write((char*) &((*sunout)(indr,indc)), sizeof(double));
                    }
                }
            }
            //now that data is printed to binary files we rezero storage matrix
            cout<<"Saving"<<endl;
            count=-1;
            (*xout).zeros();
            (*tout).zeros();
            (*hout).zeros();
            (*sunout).zeros();
            cout<<files<<endl;
            files++;
        }

        if(*inputs.flyby_toggle==1){
            f0_hyp = kepler(inputs.n_hyp, t_temp(0,0), inputs.e_hyp, inputs.tau_hyp);
            X_s = kepler2cart(inputs.a_hyp, inputs.e_hyp, inputs.i_hyp, inputs.RAAN_hyp, inputs.om_hyp, f0_hyp, inputs.G, inputs.Mplanet);
            (*hout).row(count+1)=X_s.t();
        }
        if(*inputs.helio_toggle==1){
            f0_helio = kepler(inputs.n_helio, t_temp(0,0), inputs.e_helio, inputs.tau_helio);
            X_helio = kepler2cart(inputs.a_helio, inputs.e_helio, inputs.i_helio, inputs.RAAN_helio, inputs.om_helio, f0_helio, inputs.G, inputs.Msolar);
            (*sunout).row(count+1)=X_helio.t();
        }
        (*xout).row(count+1)=next_y;
        (*tout).row(count+1)=t_temp+h;
        count++;
        t=t_temp+h;
    }
    //now that integration loop is done write out any other output data to binary files
    if (count==0){
        if(*inputs.flyby_toggle==1){
            (*hout)=X_s.t();
        }
        (*sunout)=X_helio.t();
        (*tout)=t_temp+h;
        (*xout)=next_y;
        for (int indr=0;indr<(*xout).n_rows;indr++){
            tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
            for (int indc=0;indc<(*xout).n_cols;indc++){
                xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
            }
            if(*inputs.flyby_toggle==1){
                for (int indc=0;indc<(*hout).n_cols;indc++){
                    hfile.write((char*) &((*hout)(indr,indc)), sizeof(double));
                }
            }
            if(*inputs.helio_toggle==1){
                for (int indc=0;indc<(*sunout).n_cols;indc++){
                    sunfile.write((char*) &((*sunout)(indr,indc)), sizeof(double));
                }
            }
        }
    }
    else{
        (*xout)=(*xout)(span(0,count),span(0,29));
        (*tout)=(*tout)(span(0,count),0);
        (*hout)=(*hout)(span(0,count),span(0,5));
        (*sunout)=(*sunout)(span(0,count),span(0,5));
        for (int indr=0;indr<(*xout).n_rows;indr++){
            tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
            for (int indc=0;indc<(*xout).n_cols;indc++){
                xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
            }
            if(*inputs.flyby_toggle==1){
                for (int indc=0;indc<(*hout).n_cols;indc++){
                    hfile.write((char*) &((*hout)(indr,indc)), sizeof(double));
                }
            }
            if(*inputs.helio_toggle==1){
                for (int indc=0;indc<(*sunout).n_cols;indc++){
                    sunfile.write((char*) &((*sunout)(indr,indc)), sizeof(double));
                }
            }

        }
        
    }
    //close binary files
    xfile.close();
    tfile.close();
    hfile.close();
    sunfile.close();
    return;    
}

//This is a fixed timestep standard Adams-Bashforth-Moulton predictor-corrector integrator. The output
//is binary files of the states and time written for each evaluation of the dynamics. The dynamics used by the integrator can be changed using the ode function
//
//inputs: t0 - initial time (units of input states)
//tf - final time (units of the input states)
//x0 - row vector of the the states, the ode function must be set up to take in x0 in the form input to this function
//h - integration time step (units of input states)
//inputs - structure of parameters which may be needed by the ode function.
//tout - pointer to a matrix which will be used to house the output times
//xout - point to a matrix which will be used to house the output states
//ode - function which takes as inputs the states as x0, the time as a scalar and the inputs structure
//outputs: the integrator will save two binary files in subfolders output_t and output_x (be careful these are deleted with each run).
//the binary files are single line wrappings of the tout and xout matrix, which is to say the each row of xout is entered in line in the output binary
//the same occurs for the output time
//if the flyby or heliocentric toggles are turned on, the integrator also saves binary files for the perturbing body relative to the binary system barycenter.
//hout and sunout and pointers to a matrix to house the states of the perturber (3rd body or the sun)
void ABM(double t0, double tf, mat x0, double h, parameters inputs, mat* tout, mat* xout, mat* hout, mat*sunout, std::function<mat(mat,mat, parameters)> ode){
    //system commands which remove old integrated files and create new output directories
    system("if [ -d output_t ]; then rm -rf output_t; fi; mkdir output_t");
    system("if [ -d output_x ]; then rm -rf output_x; fi; mkdir output_x");
    system("if [ -d output_h ]; then rm -rf output_h; fi; mkdir output_h");
    system("if [ -d output_sun ]; then rm -rf output_sun; fi; mkdir output_sun");
    ofstream tfile("output_t/t_out.bin", std::ios::binary);
    ofstream xfile("output_x/x_out.bin", std::ios::binary);
    ofstream hfile("output_h/h_out.bin", std::ios::binary);
    ofstream sunfile("output_sun/sun_out.bin", std::ios::binary);
    //set up rk4 variables and k's for rk computation
    mat k1,k2,k3,k4,y0,y,next_y,y_0,y_1,y_2,y_3,f_0,f_1,f_2,f_3,f_p;
    mat t(1,1);
    t<<t0<<endr;
    y0=x0;
    //matrices are initialized to a limitted size. As the integrator reaches this limited size (the number of integration steps to reach one day in seconds)
    //it will write out the contents to the output binary's then zero out the matrix and begin refilling it
    //this is done to avoid large sets of data being manipulated in memory for long integrations
    (*xout).resize(int(3600*24/h),y0.n_cols);
    (*xout).row(0)=y0;
    (*tout).resize(int(3600*24/h),1);
    (*tout).row(0)=t(0,0);

    double f0_hyp = kepler(inputs.n_hyp, t0, inputs.e_hyp, inputs.tau_hyp);
    vec X_s = kepler2cart(inputs.a_hyp, inputs.e_hyp, inputs.i_hyp, inputs.RAAN_hyp, inputs.om_hyp, f0_hyp, inputs.G, inputs.Mplanet);
    (*hout).resize(int(3600*24/h),X_s.n_rows);
    (*hout).row(0)=X_s.t();

    double f0_helio = kepler(inputs.n_helio, t0, inputs.e_helio, inputs.tau_helio);
    vec X_helio = kepler2cart(inputs.a_helio, inputs.e_helio, inputs.i_helio, inputs.RAAN_helio, inputs.om_helio, f0_helio, inputs.G, inputs.Msolar);
    (*sunout).resize(int(3600*24/h),X_helio.n_rows);
    (*sunout).row(0)=X_helio.t();

    int count=0;
    int files=1;
    mat t_temp,y_temp,y_pred,t_0,t_1,t_2,t_3;
    //begin integration
    while(t(0,0)<tf){
        if(t(0,0)<=3*h){
            t_temp=(*tout)(count,0);
            y_temp=(*xout).row(count);
            //rk4 steps
            k1=ode(y_temp,t_temp,inputs);
            k2=ode(y_temp+(h/2.)*k1.t(),t_temp+(h/2.),inputs);
            k3=ode(y_temp+(h/2.)*k2.t(),t_temp+(h/2.),inputs);
            k4=ode(y_temp+h*k3.t(),t_temp+h,inputs);
            //rk calculation
            next_y=y_temp+(h/6.)*(k1.t()+2.*k2.t()+2.*k3.t()+k4.t());
            (*xout).row(count+1)=next_y;
            (*tout).row(count+1)=t_temp+h;
            count++;
            t=t_temp+h;
        }
        else{
            t_0=(*tout)(count,0);
            y_0=(*xout).row(count);
            f_0=ode(y_0,t_0,inputs);
            //Predictor steps
            if(t(0,0)<=4*h){
                t_1=(*tout)(count-1,0);
                y_1=(*xout).row(count-1);
                t_2=(*tout)(count-2,0);
                y_2=(*xout).row(count-2);
                t_3=(*tout)(count-3,0);
                y_3=(*xout).row(count-3);
                f_1=ode(y_1,t_1,inputs);
                f_2=ode(y_2,t_2,inputs);
                f_3=ode(y_3,t_3,inputs);
            }
            //Predictor Calculation
            y_pred = y_0 + (h/24.)*(55.*f_0.t()-59.*f_1.t()+37.*f_2.t()-9.*f_3.t());
            //Corrector calculation
            f_p=ode(y_pred,t_0+h,inputs);
            next_y = y_0 + (h/24.)*(9.*f_p.t()+19.*f_0.t()-5.*f_1.t()+f_2.t());
            if (count==(int(86400/h)-1)){//this is the print out to the binary files
                for (int indr=0;indr<(*xout).n_rows;indr++){
                    tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
                    for (int indc=0;indc<(*xout).n_cols;indc++){
                        xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
                    }
                    if(*inputs.flyby_toggle==1){
                        for (int indc=0;indc<(*hout).n_cols;indc++){
                            hfile.write((char*) &((*hout)(indr,indc)), sizeof(double));
                        }
                    }
                    if(*inputs.helio_toggle==1){
                        for (int indc=0;indc<(*sunout).n_cols;indc++){
                            sunfile.write((char*) &((*sunout)(indr,indc)), sizeof(double));
                        }
                    }
                }
                //now that data is printed to binary files we rezero storage matrix
                cout<<"Saving"<<endl;
                count=-1;
                (*xout).zeros();
                (*tout).zeros();
                (*hout).zeros();
                (*sunout).zeros();
                cout<<files<<endl;
                files++;
            }
        if(*inputs.flyby_toggle==1){    
            f0_hyp = kepler(inputs.n_hyp, t_0(0,0), inputs.e_hyp, inputs.tau_hyp);
            X_s = kepler2cart(inputs.a_hyp, inputs.e_hyp, inputs.i_hyp, inputs.RAAN_hyp, inputs.om_hyp, f0_hyp, inputs.G, inputs.Mplanet);
            (*hout).row(count+1)=X_s.t();
        }
        if(*inputs.helio_toggle==1){ 
            f0_helio = kepler(inputs.n_helio, t_0(0,0), inputs.e_helio, inputs.tau_helio);
            X_helio = kepler2cart(inputs.a_helio, inputs.e_helio, inputs.i_helio, inputs.RAAN_helio, inputs.om_helio, f0_helio, inputs.G, inputs.Msolar);
            (*sunout).row(count+1)=X_helio.t();
        }
        (*xout).row(count+1)=next_y;
        (*tout).row(count+1)=t_0+h;
        count++;
        t=t_0+h;
        f_3=f_2;
        f_2=f_1;
        f_1=f_0;
        }
    }
    //now that integration loop is done write out any other output data to binary files
    if (count==0){
        (*tout)=t_temp+h;
        (*xout)=next_y;
        (*hout)=X_s.t();
        (*sunout)=X_helio.t();
        for (int indr=0;indr<(*xout).n_rows;indr++){
            tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
            for (int indc=0;indc<(*xout).n_cols;indc++){
                xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
            }
            if(*inputs.flyby_toggle==1){
                for (int indc=0;indc<(*hout).n_cols;indc++){
                     hfile.write((char*) &((*hout)(indr,indc)), sizeof(double));
                }
            }
            if(*inputs.helio_toggle==1){
                for (int indc=0;indc<(*sunout).n_cols;indc++){
                    sunfile.write((char*) &((*sunout)(indr,indc)), sizeof(double));
                }
            }
        }
    }
    else{
        (*xout)=(*xout)(span(0,count),span(0,29));
        (*tout)=(*tout)(span(0,count),0);
        (*hout)=(*hout)(span(0,count),span(0,5));
        (*sunout)=(*sunout)(span(0,count),span(0,5));
        for (int indr=0;indr<(*xout).n_rows;indr++){
            tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
            for (int indc=0;indc<(*xout).n_cols;indc++){
                xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
            }
            if(*inputs.flyby_toggle==1){
                for (int indc=0;indc<(*hout).n_cols;indc++){
                    hfile.write((char*) &((*hout)(indr,indc)), sizeof(double));
                }
            }
            if(*inputs.helio_toggle==1){
                for (int indc=0;indc<(*sunout).n_cols;indc++){
                    sunfile.write((char*) &((*sunout)(indr,indc)), sizeof(double));
                }
            }
        }
        
    }
    //close binary files
    xfile.close();
    tfile.close();
    hfile.close();
    sunfile.close();
    return;    
}

// LGVI integrator function implemented using the "hamiltonian_map" function to propogate the dynamics. The integrator aims to conserve SO(3) and
// is symplectic. This implementation is a fixed step integrator and simply steps the hamiltonian map through an evaluation at each time step
//
//inputs: t0 - initial time (units of input states)
//tf - final time (units of the input states)
//x0 - row vector of the the initial states
//h - integration time step (units of input states)
//inputs - structure of parameters which may be needed by the ode function.
//tout - pointer to a matrix which will be used to house the output times
//xout - point to a matrix which will be used to house the output states
//outputs: the integrator will save two binary files in subfolders output_t and output_x (be careful these are deleted with each run).
//the binary files are single line wrappings of the tout and xout matrix, which is to say the each row of xout is entered in line in the output binary
//the same occurs for the output time
void LGVI_integ(double t0, double tf, mat x0, double h, parameters inputs, mat* tout, mat* xout){
    //system commands which remove old integrated files and create new output directories
    system("if [ -d output_t ]; then rm -rf output_t; fi; mkdir output_t");
    system("if [ -d output_x ]; then rm -rf output_x; fi; mkdir output_x");
    ofstream tfile("output_t/t_out.bin", std::ios::binary);
    ofstream xfile("output_x/x_out.bin", std::ios::binary);
    // setup variables for propogation
    mat y0,y,next_y;
    mat t(1,1);
    t<<t0<<endr;
    y0=x0;
    //matrices are initialized to a limitted size. As the integrator reaches this limited size (the number of integration steps to reach one day in seconds)
    //it will write out the contents to the output binary's then zero out the matrix and begin refilling it
    //this is done to avoid large sets of data being manipulated in memory for long integrations
    (*xout).resize(int(3600*24/h),y0.n_cols);
    (*xout).row(0)=y0;
    (*tout).resize(int(3600*24/h),1);
    (*tout).row(0)=t(0,0);
    int count=0;
    int files=1;
    //Define modified moments of inertia
    (*(inputs.IdA))=2.*(.5*trace(diagmat((*(inputs.IA))))*eye(3,3)-diagmat(*(inputs.IA)));
    (*(inputs.IdB))=2.*(.5*trace(diagmat((*(inputs.IB))))*eye(3,3)-diagmat(*(inputs.IB)));
    //Create matrices for SO(3) implicit equations used by LGVI
    mat fab(3,1), fna(3,1), Fab(3,3), Fna(3,3), g_map(3,1), G_map(3,1), grad_G(3,3);
    fab.fill(0.4);
    fna.fill(0.4);
    //Initialize dynamical parameters and potential evaluations
    mat t_temp,y_temp;
    mat du_dr_0(3,1), M_0(3,1);
    mat r0=x0.cols(0,2);
    mat C0=x0.cols(21,29);
    r0.reshape(3,1);
    C0.reshape(3,3);
    C0=C0.t();
    map_potential_partials(&C0, &r0, inputs, &du_dr_0, &M_0);
    //begin integration
    while(t(0,0)<tf){
        t_temp=(*tout)(count,0);
        y_temp=(*xout).row(count);
        next_y=zeros(1,y0.n_cols);
        //LGVI hamiltonian map evaluation
        hamiltonian_map(h, inputs, &y_temp, &next_y, &fab, &fna, &Fab, &Fna, &g_map, &G_map, &grad_G, &du_dr_0, &M_0, &x0);
        if (count==(int(86400/h)-1)){//this is the print out to the binary files
            for (int indr=0;indr<(*xout).n_rows;indr++){
                tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
                for (int indc=0;indc<(*xout).n_cols;indc++){
                    xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
                }
            }
            //now that data is printed to binary files we rezero storage matrix
            cout<<"Saving"<<endl;
            count=-1;
            (*xout).zeros();
            (*tout).zeros();
            cout<<files<<endl;
            files++;
        }
        (*xout).row(count+1)=next_y;
        (*tout).row(count+1)=t_temp+h;
        count++;
        t=t_temp+h;
    }
    //now that integration loop is done write out any other output data to binary files
    if (count==0){
        (*tout)=t_temp+h;
        (*xout)=next_y;
        for (int indr=0;indr<(*xout).n_rows;indr++){
            tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
            for (int indc=0;indc<(*xout).n_cols;indc++){
                xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
            }
        }
    }
    else{
        (*xout)=(*xout)(span(0,count),span(0,29));
        (*tout)=(*tout)(span(0,count),0);
        for (int indr=0;indr<(*xout).n_rows;indr++){
            tfile.write((char*) &((*tout)(indr,0)), sizeof(double));
            for (int indc=0;indc<(*xout).n_cols;indc++){
                xfile.write((char*) &((*xout)(indr,indc)), sizeof(double));
            }
        }
        
    }
    //close binary files
    xfile.close();
    tfile.close();
 
    return;
}

// LGVI hamiltonian map function used to evaluate each time step of the LGVI. The dynamics variables passed in are normalized by position, mass and time
// using the initial conditions for intermediate calculations and then denormalized at the end of the evaluation
//
//inputs: h - time step (units of input states)
//inputs - structure of parameters which may be needed by the ode function.
//x - pointer to row vector of the previous states
//x_out - pointer to a matrix which will be used to house the output states
//fab - pointer to relative attitude implicit solver column vector
//fna - pointer to inertial attitude implicit solver column vector
//Fab - pointer to relative attitude implicit solver rotation matrix
//Fna - pointer to inertial attitude implicit solver rotation matrix
//g_map - pointer to dynamics vector for attitude implicit solver
//G_map - pointer to implicit matrix for attitude implicit solver
//grad_G - pointer for gradient of implicit matrix for attitude implicit solver
//du_dr_n - pointer to previous evaluation of mutual potential partial with respect to position
//M_n - pointer to previous evaluation of gravity torques
//x0 - pointer to initial states row vector of simulation used to normalize intermediate calculations.
//outputs: the integrator will save two binary files in subfolders output_t and output_x (be careful these are deleted with each run).
//the binary files are single line wrappings of the tout and xout matrix, which is to say the each row of xout is entered in line in the output binary
//the same occurs for the output time
void hamiltonian_map(double h, parameters inputs, mat* x, mat* x_out, mat* fab, mat* fna, mat* Fab, mat* Fna, mat* g_map, mat* G_map, mat* grad_G, mat* du_dr_n, mat* M_n, mat* x0){
    //all inputs are in A frame
    //compute normalizing constants using simulation initial conditions
    double nr=norm((*x0).cols(0,2));
    double nm=(*(inputs.m));
    double nt=sqrt(*(inputs.G)*nm/pow(nr,3));
    //unpack previous states and normalize values
    h=h*nt;
    mat r=(*x).cols(0,2)/nr;
    mat v=(*x).cols(3,5)/nr/nt;
    mat wc=(*x).cols(6,8)/nt;
    mat ws=(*x).cols(9,11)/nt;
    mat Cc=(*x).cols(12,20);
    mat C=(*x).cols(21,29);
    r.reshape(3,1);
    v.reshape(3,1);
    wc.reshape(3,1);
    ws.reshape(3,1);
    Cc.reshape(3,3);
    Cc=Cc.t();//reshape is columnwise here so have to transpose to fix
    C.reshape(3,3);
    C=C.t();
    mat ws_s=C.t()*ws;
    //initialize new evaluations of mutual potential partials and gravity torques
    mat du_dr_n1(3,1), M_n1(3,1), du_dalpha(3,1),du_dbeta(3,1),du_dgam(3,1);
    mat r_n1, C_n1, v_n1, wc_n1, ws_n1, Cc_n1;
    //prepare and execute relative attitude implicit evaluation
    mat IBr=C*diagmat(*(inputs.IB))*C.t()/(nm*pow(nr,2.));
    (*g_map)=C*diagmat(*(inputs.IB))*C.t()*ws/(nm*pow(nr,2.)) - (h/2.)*(*M_n)/(nm*pow(nr,2.)*pow(nt,2.));
    mat IdB_rot=C*(*(inputs.IdB))*C.t()/(nm*pow(nr,2.));
    F_cayley_calc(h, g_map, &IBr, fab, Fab, G_map, grad_G);
    //prepare and execute inertial attitude implicit evaluation
    (*g_map)=diagmat(*(inputs.IA))*wc/(nm*pow(nr,2.)) + (h/2.)*cross(r,(*du_dr_n)/(nm*nr*pow(nt,2.))) + (h/2.)*(*M_n)/(nm*pow(nr,2.)*pow(nt,2.));
    mat IH=diagmat(*(inputs.IA))/(nm*pow(nr,2.));
    F_cayley_calc(h, g_map, &IH, fna, Fna, G_map, grad_G);
    //map position and relative attitude forward
    r_n1=(*Fna).t()*(r+h*v-nm*(pow(h,2.)/(2.* (*(inputs.m))))*(*du_dr_n)/(nr*nm*pow(nt,2.)));
    C_n1=(*Fna).t()*(*Fab)*C;
    //evaluate new potential partials and gravity torques
    mat r_n1n=r_n1*nr;//potential evaluation needs non-normalized inputs so create this intermediate vector
    map_potential_partials(&C_n1, &r_n1n, inputs, &du_dr_n1, &M_n1); 
    //map velocities and inertial attitude forward
    v_n1=(*Fna).t()*(v-nm*(h/(2.* (*(inputs.m))))*(*du_dr_n)/(nr*nm*pow(nt,2.)))-nm*(h/(2.* (*inputs.m)))*du_dr_n1/(nr*nm*pow(nt,2.));
    wc_n1=inv(diagmat(*(inputs.IA))/(nm*pow(nr,2.)))*((*Fna).t()*(diagmat(*(inputs.IA))/(nm*pow(nr,2.))*wc+(h/2.)*cross(r,(*du_dr_n)/(nr*nm*pow(nt,2.)))+(h/2.)*(*M_n)/(nm*pow(nr,2.)*pow(nt,2.)))
        +(h/2.)*cross(r_n1,du_dr_n1/(nr*nm*pow(nt,2.)))+(h/2.)*M_n1/(nm*pow(nr,2.)*pow(nt,2.)));
    ws_n1=C_n1*inv(diagmat(*(inputs.IB))/(nm*pow(nr,2.)))*C_n1.t()*((*Fna).t()*(C*(diagmat(*(inputs.IB))/(nm*pow(nr,2.)))*C.t()*ws-(h/2.)*(*M_n)/(nm*pow(nr,2.)*pow(nt,2.)))-(h/2.)*M_n1/(nm*pow(nr,2.)*pow(nt,2.)));
    Cc_n1=Cc*(*Fna);
    //repack and de-normalize new states
    r_n1=r_n1n;
    v_n1=v_n1*nr*nt;
    ws_n1=ws_n1*nt;
    wc_n1=wc_n1*nt;
    (*x_out)<<r_n1(0,0)<<r_n1(1,0)<<r_n1(2,0)<<v_n1(0,0)<<v_n1(1,0)<<
            v_n1(2,0)<<wc_n1(0,0)<<wc_n1(1,0)<<wc_n1(2,0)<<
            ws_n1(0,0)<<ws_n1(1,0)<<ws_n1(2,0)<<Cc_n1(0,0)<<
            Cc_n1(0,1)<<Cc_n1(0,2)<<Cc_n1(1,0)<<Cc_n1(1,1)<<
            Cc_n1(1,2)<<Cc_n1(2,0)<<Cc_n1(2,1)<<Cc_n1(2,2)<<
            C_n1(0,0)<<C_n1(0,1)<<C_n1(0,2)<<C_n1(1,0)<<
            C_n1(1,1)<<C_n1(1,2)<<C_n1(2,0)<<C_n1(2,1)<<
            C_n1(2,2)<<endr;
    //store new evaluations of potential partials and gravity torques for use in next evaluation
    (*du_dr_n)=du_dr_n1;
    (*M_n)=M_n1;
    return;
}

// F_cayley_calc function implements the cayley implicit solver for the LGVI described in Lee 2007 LGVI paper
//
//inputs: h - time step normalized
//g - pointer to dynamics vector for attitude implicit solver
//I - pointer to inertia
//f - pointer to attitude implicit solver column vector
//F - pointer to attitude implicit solver rotation matrix
//G - pointer to implicit matrix for attitude implicit solver
//grad_G - pointer for gradient of implicit matrix for attitude implicit solver
//outputs: saves the converged variables
void F_cayley_calc(double h, mat* g, mat* I, mat* f, mat* F, mat* G, mat* grad_G){
    (*f).fill(.0001);
    (*G) = ones(3,1);
    (*g)=h*(*g);
    int check=0;
    while (norm(*G)>(1.e-15)){
        (*G) =(*g) + cross(*g,*f) + dot(*g,*f) * (*f) - 2.*(*I)*(*f);
        (*grad_G) = tilde_opv(*g) + dot(*g,*f)*eye(3,3) + (*f)*(*g).t() -2.*(*I);
        (*f) = (*f) + inv(*grad_G) * (- (*G));
        check++;
        //if the cayley solver isn't working switch to the exponential solver  and compare accuracy
        if (check>100){
            cout<<"broke Cayley"<<endl;
            double cay_err=norm(*G);
            mat cay_f=(*f);
            F_exp_calc(h, g, I, f, F, G, grad_G);
            if (norm((*g) - (*G))>cay_err){
                (*f)=cay_f;
            }
            break;
        }
    }
    //compute rotation matrix
    if (check<=100){
        (*F) = (eye(3,3) + tilde_opv(*f)) * inv(eye(3,3) - tilde_opv(*f));
    }
    return;
}

// F_exp_calc function implements the exponential implicit solver for the LGVI described in Lee 2007 LGVI paper. Exponential solver
// is only used if cayley implicit solver has failed to converge
//
//inputs: h - time step normalized
//g - pointer to dynamics vector for attitude implicit solver
//I - pointer to inertia
//f - pointer to attitude implicit solver column vector
//F - pointer to attitude implicit solver rotation matrix
//G - pointer to implicit matrix for attitude implicit solver
//grad_G - pointer for gradient of implicit matrix for attitude implicit solver
//outputs: saves the converged variables
void F_exp_calc(double h, mat* g, mat* I, mat* f, mat* F, mat* G, mat* grad_G){
    int check=0;
    //intialize implcit solver
    (*f).fill(.1);
    (*G) = ones(3,1);
    //begin implicit loop
    while (norm((*g) - (*G))>(1.e-15)){
        (*G)=sin(norm(*f))*(*I)*(*f)/norm(*f) +(1-cos(norm(*f)))*cross(*f,(*I)*(*f))/pow(norm(*f),2);
        (*grad_G)=(norm(*f)*cos(norm(*f))-sin(norm(*f)))*(*I)*(*f)*(*f).t()/pow(norm(*f),3)+sin(norm(*f))*(*I)/norm(*f)+(norm(*f)*sin(norm(*f))-2.*(1.-cos(norm(*f))))*cross(*f,(*I)*(*f))*(*f).t()/pow(norm(*f),4)+(1.-cos(norm(*f)))*(-tilde_opv((*I)*(*f))+tilde_opv(*f)*(*I))/pow(norm(*f),2);
        (*f)=(*f)+inv(*grad_G)*((*g)-(*G));
        check++;
        //check for failure warning if exponential solver fails (exponential solver is only used when cayley sover has failed)
        if (check>100){
            cout<<"Warning: Computation of F matrix did not converge to numerical precision."<<endl;
            break;
        }
    }
    //compute rotation matrix
    (*F)=eye(3,3)+sin(norm(*f))*tilde_opv(*f)/norm(*f)+(1.-cos(norm(*f)))*tilde_opv(*f)*tilde_opv(*f)/pow(norm(*f),2);
    return;
}

// map_potential_partials is used to compute the potential partials and gravity torques for each LGVI time step evaluations.
// The LGVI computations are done using normalized variables, but this requires non-normalized variables so variables must be modified
// before being passed to this function
//
//inputs: C - pointer to relative rotation matrix
//r - pointer to relative position vector
//inputs - structure of parameters which may be needed
//du_dr - pointer to mutual gravity potential partial with repect to position
//M - pointer to gravity torques
//outputs: no outputs, pointers for potential partials and gravity torques are used to store new values for the LGVI
void map_potential_partials(mat* C, mat* r, parameters inputs, mat* du_dr, mat* M){
    //initialize states and potential partials
    double R=norm(*r,2);
    mat e=(*r).t()/R;
    mat du_dalpha(3,1), du_dbeta(3,1), du_dgam(3,1);
    //rotate secondary inertia integrals
    inertia_rot(*C, *(inputs.n), inputs.TB, inputs.TBp);
    //compute partials of rotated secondary inertia integrals with respect to each element of C
    dT_dc(0, 0, *C, *(inputs.n), inputs.TB, &((*(inputs.dT))(0,0)));
    dT_dc(0, 1, *C, *(inputs.n), inputs.TB, &((*(inputs.dT))(0,1)));
    dT_dc(0, 2, *C, *(inputs.n), inputs.TB, &((*(inputs.dT))(0,2)));
    dT_dc(1, 0, *C, *(inputs.n), inputs.TB, &((*(inputs.dT))(1,0)));
    dT_dc(1, 1, *C, *(inputs.n), inputs.TB, &((*(inputs.dT))(1,1)));
    dT_dc(1, 2, *C, *(inputs.n), inputs.TB, &((*(inputs.dT))(1,2)));
    dT_dc(2, 0, *C, *(inputs.n), inputs.TB, &((*(inputs.dT))(2,0)));
    dT_dc(2, 1, *C, *(inputs.n), inputs.TB, &((*(inputs.dT))(2,1)));
    dT_dc(2, 2, *C, *(inputs.n), inputs.TB, &((*(inputs.dT))(2,2)));
    //compute dU_dr
    (*du_dr)(0,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e, R, 0, inputs.TA, inputs.TBp);
    (*du_dr)(1,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e, R, 1, inputs.TA, inputs.TBp);
    (*du_dr)(2,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e, R, 2, inputs.TA, inputs.TBp);
    //compute partial of U with respect to first column of C for torque calc
    du_dalpha(0,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(0,0)));
    du_dalpha(1,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(1,0)));
    du_dalpha(2,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(2,0)));
    //compute partial of U with respect to second column of C for torque calc
    du_dbeta(0,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(0,1)));
    du_dbeta(1,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(1,1)));
    du_dbeta(2,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(2,1)));
    ////compute partial of U with respect to third column of C for torque calc
    du_dgam(0,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(0,2)));
    du_dgam(1,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(1,2)));
    du_dgam(2,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(2,2)));
    //compute torques
    (*M)=cross((*C).col(0),du_dalpha)
           + cross((*C).col(1),du_dbeta)
            +cross((*C).col(2),du_dgam);
    return;
}

//Continuous Equations of motion implemented from Maciejewski 1995 as modeled after the realization specified in Hou 2016
//This is the ODE file used for the RK 4, RK 7(8), and ABM. Units of km kg and sec
//inputs: x - row vector of input states specified in A frame with order [r,v,wc,ws,Cc,C], rotation matrices assumed to be row wrapped
//t - time seconds, this parameter is not used
//parameters inputs - structure defined at top of this code and commented in detail
//outputs: xd - RHS of equations of motion in same order as as but [rd,vd,wcd,wsd,Ccd,Cd] where d denotes rate of change
mat hou_ode(mat x, mat t, parameters inputs){
    //all inputs are in A frame
    //unpack states
    mat r=x.cols(0,2);
    mat v=x.cols(3,5);
    mat wc=x.cols(6,8);
    mat ws=x.cols(9,11);
    mat Cc=x.cols(12,20);
    mat C=x.cols(21,29);
    double time = t(0,0);
    r.reshape(3,1);
    v.reshape(3,1);
    wc.reshape(3,1);
    ws.reshape(3,1);
    Cc.reshape(3,3);
    Cc=Cc.t();//reshape is columnwise here so have to transpose to fix
    C.reshape(3,3);
    C=C.t();
    //set up rotated states and inertias
    mat Ic_inv(3,3),Is_inv(3,3);
    double R=norm(r,2);//used for potential
    mat e=r.t()/R;//used for potential, must be row vec
    Ic_inv<<1./ (*(inputs.IA))(0,0)<<0.<<0.<<endr<<0.<<1./ (*(inputs.IA))(0,1)<<0.<<endr<<0.<<0.<<1./ (*(inputs.IA))(0,2)<<endr;
    Is_inv<<1./ (*(inputs.IB))(0,0)<<0.<<0.<<endr<<0.<<1./ (*(inputs.IB))(0,1)<<0.<<endr<<0.<<0.<<1./ (*(inputs.IB))(0,2)<<endr;
    mat Is_inv_c=C*Is_inv*C.t();
    mat wc_tilde=tilde_op(wc(0,0),wc(1,0),wc(2,0));
    mat ws_s=C.t()*ws;
    mat ws_tilde=tilde_op(ws_s(0,0),ws_s(1,0),ws_s(2,0));
    //rotate secondary inertia integrals
    inertia_rot(C, *(inputs.n), inputs.TB, inputs.TBp);
    //get additional force values

    // Compute 3rd body perturbation
    mat M_sa(3,1),M_sb(3,1);
    if ((*(inputs.flyby_toggle))==1){
        // True anomaly of 3rd body relative to binary barycenter
        double f0_hyp = kepler(inputs.n_hyp, time, inputs.e_hyp, inputs.tau_hyp);
        // Keplerian elements of 3rd body relative to binary barycenter
        vec X_s = kepler2cart(inputs.a_hyp, inputs.e_hyp, inputs.i_hyp, inputs.RAAN_hyp, inputs.om_hyp, f0_hyp, inputs.G, inputs.Mplanet);
        vec R_s;
        R_s<<X_s(0)<<endr<<X_s(1)<<endr<<X_s(2)<<endr;
        // 3rd body gravitational perturbation
        grav_3BP(R_s, &Cc, &r, inputs.nu, inputs.G, inputs.Mplanet, inputs.acc_3BP);
        vec R_sa = Cc.t()*R_s+(*inputs.nu)*(r);
        vec R_sb = Cc.t()*R_s-(1-(*inputs.nu))*(r);
        double Rsa_mag = norm(R_sa,2);
        double Rsb_mag = norm(R_sb,2);  
        mat e_sa = R_sa.t()/Rsa_mag;
        mat e_sb = R_sb.t()/Rsb_mag; 

        // Compute dU_dr
        mat du_drsa(3,1),du_drsb(3,1);
        du_drsa(0,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_sa, Rsa_mag, 0, inputs.TA, inputs.TS);
        du_drsa(1,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_sa, Rsa_mag, 1, inputs.TA, inputs.TS);
        du_drsa(2,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_sa, Rsa_mag, 2, inputs.TA, inputs.TS);

        du_drsb(0,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_sb, Rsb_mag, 0, inputs.TBp, inputs.TS);
        du_drsb(1,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_sb, Rsb_mag, 1, inputs.TBp, inputs.TS);
        du_drsb(2,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_sb, Rsb_mag, 2, inputs.TBp, inputs.TS);

        // Compute torques in A frame due to 3rd body perturbation
        M_sa=cross(R_sa,du_drsa);
        M_sb=cross(R_sb,du_drsb);
    }
    else{
        (*(inputs.acc_3BP)).zeros();
        M_sa.zeros();
        M_sb.zeros();
    }
    // Compute heliocentric perturbation
    mat M_suna(3,1),M_sunb(3,1);
    if ((*(inputs.helio_toggle))==1){
        // True anomaly of sun relative to binary barycenter
        double f0_helio = kepler(inputs.n_helio, time, inputs.e_helio, inputs.tau_helio);
        // Keplerian elements of sun relative to binary barycenter
        vec X_sun = kepler2cart(inputs.a_helio, inputs.e_helio, inputs.i_helio, inputs.RAAN_helio, inputs.om_helio, f0_helio, inputs.G, inputs.Msolar);
        vec R_sun;
        R_sun<<X_sun(0)<<endr<<X_sun(1)<<endr<<X_sun(2)<<endr;
        solar_accel(R_sun, &Cc, &r, inputs.nu, inputs.G, inputs.Msolar, inputs.acc_solar);
        vec R_suna = Cc.t()*R_sun+(*inputs.nu)*(r);
        vec R_sunb = Cc.t()*R_sun-(1-(*inputs.nu))*(r);
        double Rsuna_mag = norm(R_suna,2);
        double Rsunb_mag = norm(R_sunb,2);  
        mat e_suna = R_suna.t()/Rsuna_mag;
        mat e_sunb = R_sunb.t()/Rsunb_mag; 

        // Compute dU_dr
        mat du_drsuna(3,1),du_drsunb(3,1);
        du_drsuna(0,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_suna, Rsuna_mag, 0, inputs.TA, inputs.Tsun);
        du_drsuna(1,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_suna, Rsuna_mag, 1, inputs.TA, inputs.Tsun);
        du_drsuna(2,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_suna, Rsuna_mag, 2, inputs.TA, inputs.Tsun);

        du_drsunb(0,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_sunb, Rsunb_mag, 0, inputs.TBp, inputs.Tsun);
        du_drsunb(1,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_sunb, Rsunb_mag, 1, inputs.TBp, inputs.Tsun);
        du_drsunb(2,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e_sunb, Rsunb_mag, 2, inputs.TBp, inputs.Tsun);

        // Compute torques in A frame due to solar gravity perturbation
        M_suna=cross(R_suna,du_drsuna);
        M_sunb=cross(R_sunb,du_drsunb);
    }
    else{
        (*(inputs.acc_solar)).zeros();
        M_suna.zeros();
        M_sunb.zeros();
    }
    // Legacy heliocentric perturbation
    if ((*(inputs.sg_toggle))==1){
        hill_solar_grav(&Cc, &r, &v, inputs.mean_motion, inputs.sg_acc);
    }
    else{
        (*(inputs.sg_acc)).zeros();
    }
    // Compute energy dissipation due to gravitational tides between the primary and secondary
    if (*(inputs.tt_toggle)==1){
        md_tidal_torque(&r, &v, &wc, &ws, &Cc, &C, inputs);
    }
    else{
        (*(inputs.tt_1)).zeros();
        (*(inputs.tt_2)).zeros();
        (*(inputs.tt_orbit)).zeros();
    }
    
    //compute partials of rotated secondary inertia integrals with respect to each element of C
    dT_dc(0, 0, C, *(inputs.n), inputs.TB, &((*(inputs.dT))(0,0)));
    dT_dc(0, 1, C, *(inputs.n), inputs.TB, &((*(inputs.dT))(0,1)));
    dT_dc(0, 2, C, *(inputs.n), inputs.TB, &((*(inputs.dT))(0,2)));
    dT_dc(1, 0, C, *(inputs.n), inputs.TB, &((*(inputs.dT))(1,0)));
    dT_dc(1, 1, C, *(inputs.n), inputs.TB, &((*(inputs.dT))(1,1)));
    dT_dc(1, 2, C, *(inputs.n), inputs.TB, &((*(inputs.dT))(1,2)));
    dT_dc(2, 0, C, *(inputs.n), inputs.TB, &((*(inputs.dT))(2,0)));
    dT_dc(2, 1, C, *(inputs.n), inputs.TB, &((*(inputs.dT))(2,1)));
    dT_dc(2, 2, C, *(inputs.n), inputs.TB, &((*(inputs.dT))(2,2)));
    
    //compute dU_dr
    mat du_dr(3,1),du_dalpha(3,1),du_dbeta(3,1),du_dgam(3,1);
    du_dr(0,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e, R, 0, inputs.TA, inputs.TBp);
    du_dr(1,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e, R, 1, inputs.TA, inputs.TBp);
    du_dr(2,0)=du_x(*(inputs.G), *(inputs.n),inputs.tk, inputs.a, inputs.b, e, R, 2, inputs.TA, inputs.TBp);
    //compute partial of U with respect to first column of C for torque calc
    du_dalpha(0,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(0,0)));
    du_dalpha(1,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(1,0)));
    du_dalpha(2,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(2,0)));
    //compute partial of U with respect to second column of C for torque calc
    du_dbeta(0,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(0,1)));
    du_dbeta(1,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(1,1)));
    du_dbeta(2,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(2,1)));
    ////compute partial of U with respect to third column of C for torque calc
    du_dgam(0,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(0,2)));
    du_dgam(1,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(1,2)));
    du_dgam(2,0)=du_c(*(inputs.G), *(inputs.n), inputs.tk, inputs.a, inputs.b, e, R, inputs.TA, &((*(inputs.dT))(2,2)));
    //compute angular momentum in A frame
    mat La=diagmat(*(inputs.IA))*wc;
    mat Lb=C*diagmat(*(inputs.IB))*ws_s;
    //compute torques
    mat Mb=-cross(C.col(0),du_dalpha)
            -cross(C.col(1),du_dbeta)
            -cross(C.col(2),du_dgam);
    mat Ma=cross(r,du_dr)-Mb;
    //equations of motion all in A frame as described in Hou 2016
    mat rd=cross(r,wc)+v;
    mat vd=cross(v,wc)-(1./ *(inputs.m))*du_dr+(*(inputs.sg_acc))+(*(inputs.acc_3BP))+(*(inputs.acc_solar))+(*(inputs.tt_orbit));
    mat Cdc=Cc*wc_tilde;
    //rate of change of relative attitude matrix C comes from difference of primary and secondary angular velocities
    mat Cd=C*ws_tilde-wc_tilde*C;
    mat CdT=-ws_tilde*C.t()+C.t()*wc_tilde;
    mat wcd=Ic_inv*(cross(La,wc)+Ma-(*(inputs.tt_1)));
    //because of use of C and Cd must accounrt for partial of rotated inertia matrix with respect to time
    mat wsd=Is_inv_c*(cross(Lb,wc)+Mb-Cd*diagmat(*(inputs.IB))*C.t()*ws
            -C*diagmat(*(inputs.IB))*CdT*ws-(*(inputs.tt_2)));
    //repack EOM results
    mat xd(30,1);
    xd<<rd(0,0)<<endr<<rd(1,0)<<endr<<rd(2,0)<<endr<<vd(0,0)<<endr<<vd(1,0)<<
            endr<<vd(2,0)<<endr<<wcd(0,0)<<endr<<wcd(1,0)<<endr<<wcd(2,0)<<endr<<
            wsd(0,0)<<endr<<wsd(1,0)<<endr<<wsd(2,0)<<endr<<Cdc(0,0)<<endr<<
            Cdc(0,1)<<endr<<Cdc(0,2)<<endr<<Cdc(1,0)<<endr<<Cdc(1,1)<<endr<<
            Cdc(1,2)<<endr<<Cdc(2,0)<<endr<<Cdc(2,1)<<endr<<Cdc(2,2)<<endr<<
            Cd(0,0)<<endr<<Cd(0,1)<<endr<<Cd(0,2)<<endr<<Cd(1,0)<<endr<<
            Cd(1,1)<<endr<<Cd(1,2)<<endr<<Cd(2,0)<<endr<<Cd(2,1)<<endr<<
            Cd(2,2)<<endr;
    return xd;
}

// This is legacy code to calculate the effects of solar gravity. This assumes a planar circular orbit around the sun.
// For a more accurate solar perturbation use heliocentric orbit perturbation.
void hill_solar_grav(mat* NA, mat* pos, mat* vel, double* n, mat* acc){
    mat CHp_mat(3,3);
    CHp_mat.zeros();
    mat CHv_mat(3,3);
    CHv_mat.zeros();
    CHp_mat(0,0) = 3.*pow((*n),2.);
    CHv_mat(0,1) = 2.*(*n);
    CHv_mat(1,0) = -2.*(*n);
    CHp_mat(2,2) = -pow((*n),2.);
    (*acc) = (*NA).t()*(CHp_mat*(*NA)*(*pos) + CHv_mat*(*NA)*(*vel));
    return;
}

// This calculates the gravitational perturbation of a 3rd (spherical) body on the binary system
void grav_3BP(vec R_s, mat* NA, mat* pos, double* nu, double* G, double* Mplanet, mat* acc_3BP){
    R_s = (*NA).t()*R_s;
    vec R = (*pos);
    (*acc_3BP) = (*G)*(*Mplanet)*((R_s-(1-(*nu)*R))/(pow(norm(R_s-(1-(*nu)*R)),3))-(R_s+(*nu)*R)/(pow(norm(R_s+(*nu)*R),3)));
}

// This calculates the gravitational perturbation of the sun on the binary system
void solar_accel(vec R_s, mat* NA, mat* pos, double* nu, double* G, double* Msun, mat* acc_solar){
    R_s = (*NA).t()*R_s;
    vec R = (*pos);
    (*acc_solar) = (*G)*(*Msun)*((R_s-(1-(*nu)*R))/(pow(norm(R_s-(1-(*nu)*R)),3))-(R_s+(*nu)*R)/(pow(norm(R_s+(*nu)*R),3)));
}

// This calculates the tidal torque between both bodies and the orbit in order to simulate energy dissipation due to internal tidal effects
void md_tidal_torque(mat* pos, mat* vel, mat* w1, mat* w2, mat* NA, mat* AB, parameters inputs){
    mat rhat = (*NA)*(*pos)/norm(*pos);
    mat vhat = (*NA)*(*vel)/norm(*vel);
    mat rcv = (*NA)*cross(*pos,*vel);
    mat zhat = rcv/norm(rcv);
    mat wsys = rcv/pow(norm(*pos),2.);
    // Relative spin rates of each body to the orbit rate
    mat phid1 = (*NA)*(*w1) - wsys;
    mat phid2 = (*NA)*(*w2) - wsys;
    // 3 dimensional torque vectors
    mat gamma_1_vec = phid1 - (dot(phid1,rhat))*rhat;
    mat gamma_1_hat = -gamma_1_vec/norm(gamma_1_vec);
    mat gamma_2_vec = phid2 - (dot(phid2,rhat))*rhat;
    mat gamma_2_hat = -gamma_2_vec/norm(gamma_2_vec);
    mat yhat = cross(zhat,rhat)/norm(cross(zhat,rhat));
    // Calculate tidal torque, linearized about the synchronous spin rate as in Jacobson & Scheeres 2011
    double g1 = 3/2*(*(inputs.love1))*pow(3/(4*M_PI*(*(inputs.rhoA))),2)*(*(inputs.G))*pow((*(inputs.TA))(0,0,0),2)*pow((*(inputs.TB))(0,0,0),2)*sin(2*(*(inputs.eps1)))/(*(inputs.refrad1))/pow(norm(*pos),6);
    double g2 = 3/2*(*(inputs.love2))*pow(3/(4*M_PI*(*(inputs.rhoB))),2)*(*(inputs.G))*pow((*(inputs.TA))(0,0,0),2)*pow((*(inputs.TB))(0,0,0),2)*sin(2*(*(inputs.eps2)))/(*(inputs.refrad2))/pow(norm(*pos),6);
    double del1=g1*pow(6./(M_PI*(*(inputs.G))*(*(inputs.rhoA))),.5)/(*(inputs.IA))(0,2);
    double del2=g2*pow(6./(M_PI*(*(inputs.G))*(*(inputs.rhoB))),.5)/(*(inputs.IB))(0,2);
    if (norm(phid1)>abs(del1)){
        (*(inputs.tt_1))=g1*(*NA).t()*gamma_1_hat;
    }
    else{
        (*(inputs.tt_1))=norm(phid1)*pow(6./(M_PI*(*(inputs.G))*(*(inputs.rhoA))),-.5)*(*(inputs.IA))(0,2)*(*NA).t()*gamma_1_hat;
    }
    if (norm(phid2)>abs(del2)){
        (*(inputs.tt_2))=g2*(*NA).t()*gamma_2_hat;
    }
    else{
        (*(inputs.tt_2))=norm(phid2)*pow(6./(M_PI*(*(inputs.G))*(*(inputs.rhoB))),-.5)*(*(inputs.IB))(0,2)*(*NA).t()*gamma_2_hat;
    }
    // Calculate the tidal effect on the orbit to conserve angular momentum
    (*(inputs.tt_orbit)) = (1/(*(inputs.m)))*(cross((*(inputs.tt_1)+*(inputs.tt_2)),*pos))/pow(norm(*pos),2);
    return;
}

//Converts classic Keplerian elements to cartesian coordinates
vec kepler2cart(double* a_hyp, double* e_hyp, double* i_hyp, double* RAAN_hyp, double* om_hyp, double f0_hyp, double* G, double* Mplanet){
    vec X;
    vec Y;
    vec Z;
    vec X_hyp;
    double mu = *G*(*Mplanet);
    X<<1<<endr<<0<<endr<<0<<endr;
    Y<<0<<endr<<1<<endr<<0<<endr;
    Z<<0<<endr<<0<<endr<<1<<endr;
    vec n_omega=cos(*RAAN_hyp)*X+sin(*RAAN_hyp)*Y;
    vec n_perp=-cos(*i_hyp)*sin(*RAAN_hyp)*X+cos(*i_hyp)*cos(*RAAN_hyp)*Y+sin(*i_hyp)*Z;
    vec e_hat=cos(*om_hyp)*n_omega+sin(*om_hyp)*n_perp;
    vec e_perp=-sin(*om_hyp)*n_omega+cos(*om_hyp)*n_perp;
    double p=(*a_hyp)*(1-pow(*e_hyp,2));
    double r_mag=p/(1+(*e_hyp)*cos(f0_hyp));
    vec r=r_mag*(cos(f0_hyp)*e_hat+sin(f0_hyp)*e_perp);
    vec r_hat=r/r_mag;
    vec h_hat=cross(e_hat,e_perp);
    vec r_perp=cross(h_hat,r_hat)/norm(cross(h_hat,r_hat),2);
    double sin_gamma=(*e_hyp)*sin(f0_hyp)/sqrt(1+2*(*e_hyp)*cos(f0_hyp)+pow((*e_hyp),2));
    double cos_gamma=(1+(*e_hyp)*cos(f0_hyp))/sqrt(1+2*(*e_hyp)*cos(f0_hyp)+pow((*e_hyp),2));
    double v_mag=sqrt((mu/p)*(1+2*(*e_hyp)*cos(f0_hyp)+pow((*e_hyp),2)));
    vec v=v_mag*(sin_gamma*r_hat+cos_gamma*r_perp);
    X_hyp<<r(0)<<endr<<r(1)<<endr<<r(2)<<endr<<v(0)<<endr<<v(1)<<endr<<v(2)<<endr;
    return X_hyp;
}

//Solves Kepler's equation using a Newton solver to get true anomaly
double kepler(double *n_hyp, double t, double *e_hyp, double *tau_hyp){
    double M = (*n_hyp)*(t-(*tau_hyp));
    double tol = 0.001;
    double f;
    double df;
    double theta;
    if((*e_hyp)>1){  //Hyperbolic Flyby
        double H = M;
        if(abs(H)>acos(-1/(*e_hyp))){
            if(H>0){
                H=acos(-1/(*e_hyp));
            }
            else{
                H=-acos(-1/(*e_hyp));
            }
        }

        f=M-(*e_hyp)*sinh(H)+H;
        while(abs(f)>tol){
            f=M-(*e_hyp)*sinh(H)+H;
            df=-(*e_hyp)*cosh(H)+1;
            H=H-f/df;
            f=M-(*e_hyp)*sinh(H)+H;
        }
        theta=2*atan(sqrt(((*e_hyp)+1)/((*e_hyp)-1))*tanh(H/2));
    }
    else{   //Elliptical Oribt
        double E = M;

        f=E-M-(*e_hyp)*sin(E);
        while(abs(f)>tol){
            f=E-M-(*e_hyp)*sin(E);
            df=1-(*e_hyp)*cos(E);
            E=E-f/df;
            f=E-M-(*e_hyp)*sin(E);
        }
        theta=2*atan(sqrt((1+(*e_hyp))/(1-(*e_hyp)))*tan(E/2));
    }
    return theta;
} 

//Reads in "ic_inputs.txt" from python shell. Input: ics - structure detailed at top of code
void ic_read(initialization* ics){
    //set up local variable
    int order, order_a, order_b, a_shape, b_shape, Tgen, integ, flyby_toggle, helio_toggle, sg_toggle, tt_toggle;
    double G,rhoA,rhoB,aA,bA,cA,aB,bB,cB,t0,tf,h,tol,Mplanet,a_hyp,e_hyp,i_hyp,RAAN_hyp,om_hyp,tau_hyp,Msolar,a_helio,e_helio,i_helio,RAAN_helio,om_helio,tau_helio, sol_rad, au_def, love1, love2, refrad1, refrad2, eps1, eps2, Msun;
    string TAfile,TBfile,IAfile,IBfile,tfa,vfa,tfb,vfb;
    mat x0(1,30);
    string line;
    //standard io reading - all units expected in km kg sec-------##########
    ifstream myfile ("ic_input.txt");
    if (myfile.is_open()){
        getline (myfile,line);
        G=std::stod(line);//gravity parameter
        getline (myfile,line);
        order=std::stoi(line);//mutual potential truncation order
        getline (myfile,line);
        order_a=std::stoi(line);//primary inertia integrals truncation order
        getline (myfile,line);
        order_b=std::stoi(line);//secondary inertia integrals truncation order
        getline (myfile,line);
        aA=std::stod(line);//primary semi-major axis
        getline (myfile,line);
        bA=std::stod(line);//primary semi-intermediate axis
        getline (myfile,line);
        cA=std::stod(line);//primary semi-minor axis
        getline (myfile,line);
        aB=std::stod(line);//secondary semi-major axis
        getline (myfile,line);
        bB=std::stod(line);//secondary semi-inertmediate axis
        getline (myfile,line);
        cB=std::stod(line);//secondary semi-minor axis
        getline (myfile,line);
        a_shape=std::stoi(line);//primary shape flag
        getline (myfile,line);
        b_shape=std::stoi(line);//secondary shape flag
        getline (myfile,line);
        rhoA=std::stod(line);//primary density
        getline (myfile,line);
        rhoB=std::stod(line);//secondary density
        getline (myfile,line);
        t0=std::stod(line);//initial time
        getline (myfile,line);
        tf=std::stod(line);//final time
        getline (myfile,line);
        TAfile=line;//inertia integral output file for primary - not utilized
        getline (myfile,line);
        TBfile=line;//inertia integral output file for secondary - not utilized
        getline (myfile,line);
        IAfile=line;//primary moments of inertia output file - not utilized
        getline (myfile,line);
        IBfile=line;//secondary moments of inertia output file - not utilized
        getline (myfile,line);
        tfa=line;//primary tet file
        getline (myfile,line);
        vfa=line;//primary vert file
        getline (myfile,line);
        tfb=line;//secondary tet file
        getline (myfile,line);
        vfb=line;//secondary vert file
        for(int n=0;n<30;n++){//read in states [r,v,wc,ws,Cc,C]
            getline (myfile,line);
            std::size_t temp=0;
            x0(0,n)=std::stod(line);
        }
        getline (myfile,line);
        Tgen=std::stoi(line);//inertia integral generation flag
        getline (myfile,line);
        integ=std::stoi(line);//integrator selection flag
        getline (myfile,line);
        h=std::stod(line);//fixed step integrator time step
        getline (myfile,line);
        tol=std::stod(line);//adaptive step integrator tolerance
        getline (myfile,line);
        flyby_toggle=std::stoi(line);
        getline(myfile,line);
        helio_toggle=std::stoi(line);
        getline(myfile,line);
        sg_toggle=std::stoi(line);
        getline (myfile,line);
        tt_toggle=std::stoi(line);
        getline (myfile,line);
        Mplanet=std::stod(line);
        getline (myfile,line);
        a_hyp=std::stod(line);
        getline (myfile,line);
        e_hyp=std::stod(line);
        getline (myfile,line);
        i_hyp=std::stod(line);
        getline (myfile,line);
        RAAN_hyp=std::stod(line);
        getline (myfile,line);
        om_hyp=std::stod(line);
        getline (myfile,line);
        tau_hyp=std::stod(line);
        getline (myfile,line);
        Msolar=std::stod(line);
        getline (myfile,line);
        a_helio=std::stod(line);
        getline (myfile,line);
        e_helio=std::stod(line);
        getline (myfile,line);
        i_helio=std::stod(line);
        getline (myfile,line);
        RAAN_helio=std::stod(line);
        getline (myfile,line);
        om_helio=std::stod(line);
        getline (myfile,line);
        tau_helio=std::stod(line);
        getline (myfile,line);
        sol_rad=std::stod(line);
        getline (myfile,line);
        au_def=std::stod(line);
        getline (myfile,line);
        love1=std::stod(line);
        getline (myfile,line);
        love2=std::stod(line);
        getline (myfile,line);
        refrad1=std::stod(line);
        getline (myfile,line);
        refrad2=std::stod(line);
        getline (myfile,line);
        eps1=std::stod(line);
        getline (myfile,line);
        eps2=std::stod(line);
        getline (myfile,line);
        Msun=std::stod(line);
        myfile.close();
    }
    else{
        cout<<"input file error"<<endl;
    }
    //pack file into structure
    (*ics).G=G;
    (*ics).order=order;
    (*ics).order_a=order_a;
    (*ics).order_b=order_b;
    (*ics).aA=aA;
    (*ics).bA=bA;
    (*ics).cA=cA;
    (*ics).aB=aB;
    (*ics).bB=bB;
    (*ics).cB=cB;
    (*ics).a_shape=a_shape;
    (*ics).b_shape=b_shape;
    (*ics).rhoA=rhoA;
    (*ics).rhoB=rhoB;
    (*ics).t0=t0;
    (*ics).tf=tf;
    (*ics).TAfile=TAfile;
    (*ics).TBfile=TBfile;
    (*ics).IAfile=IAfile;
    (*ics).IBfile=IBfile;
    (*ics).tfa=tfa;
    (*ics).vfa=vfa;
    (*ics).tfb=tfb;
    (*ics).vfb=vfb;
    (*ics).x0=x0;
    (*ics).Tgen=Tgen;
    (*ics).integ=integ;
    (*ics).h=h;
    (*ics).tol=tol;
    (*ics).flyby_toggle=flyby_toggle;
    (*ics).helio_toggle=helio_toggle;
    (*ics).sg_toggle=sg_toggle;
    (*ics).tt_toggle=tt_toggle;
    (*ics).Mplanet=Mplanet;
    (*ics).a_hyp=a_hyp;
    (*ics).e_hyp=e_hyp;
    (*ics).i_hyp=i_hyp;
    (*ics).RAAN_hyp=RAAN_hyp;
    (*ics).om_hyp=om_hyp;
    (*ics).tau_hyp=tau_hyp;
    (*ics).Msolar=Msolar;
    (*ics).a_helio=a_helio;
    (*ics).e_helio=e_helio;
    (*ics).i_helio=i_helio;
    (*ics).RAAN_helio=RAAN_helio;
    (*ics).om_helio=om_helio;
    (*ics).tau_helio=tau_helio;
    (*ics).sol_rad=sol_rad;
    (*ics).au_def=au_def;
    (*ics).love1=love1;
    (*ics).love2=love2;
    (*ics).refrad1=refrad1;
    (*ics).refrad2=refrad2;
    (*ics).eps1=eps1;
    (*ics).eps2=eps2;
    (*ics).Msun=Msun;
    return;
}

//computes skew symmetric matrix based on vector elements x y and z
mat tilde_op(double x, double y, double z){
    mat tilde(3,3);
    tilde<<0.<<-z<<y<<endr<<z<<0.<<-x<<endr<<-y<<x<<0.<<endr;
    return tilde;
}
//computes skew symmetric matrix base on row vector
mat tilde_opv(mat A){
    mat tilde(3,3);
    tilde<<0.<<-A(2,0)<<A(1,0)<<endr<<A(2,0)<<0.<<-A(0,0)<<endr<<-A(1,0)<<A(0,0)<<0.<<endr;
    return tilde;
}
//computes factorial with doubles
double factorial(double x){
    double fact=1.;
    while (x>1){
        fact*=(double)x;
        x--;
    }
    return fact;
}

