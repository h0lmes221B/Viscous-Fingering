/* Programme of Viscous fingering */
/* applicable to the pure viscous fingering.							*/

/*To compile, type:
  gcc -O3 -o a.out ***.c -lm -lfftw -lrfftw
  gcc -O3 -o a.out ***.c -lm -lfftw -lrfftw
  To run the programme type "./a.out data rout"

  This produce a family of output files "rout_Suffix" which
  contain the various informations on the run.

  For cluster

  icc -O3 -tpp7 -xP -rcd -o a.out ****.c -I./myfftw/include ./myfftw/lib/librfftw.a ./myfftw/lib/libfftw.a
/*programme for postive and negtive R  */

#include <stdlib.h>
#include </usr/local/cuda/include/cufftw.h>
#include </usr/local/cuda/include/cufft.h>

#include </usr/local/cuda/include/cuda_device_runtime_api.h>

#include <stdio.h>
//#include <cufftw.h>
//#include <rfftw.h>
#include <math.h>
#include <string.h>
#include "/usr/local/cuda/include/cuda_runtime.h"
#include "/usr/local/cuda/include/device_launch_parameters.h"
#define PURE_DF  1

#define CELX 8192  		/* Number of grids or meshs according to x */
#define CELY 128       /* Number of grids or meshs according to y */
#define Lx   32768     /* length of the H-S cell */
#define Ly   512		/* width of the H-S cell equal to the Pe */

#define M_PI_2	1.57079632679489661923
#define M_2PI	6.2831853071795864769

#define tfinv cufftExecC2R
#define tf cufftExecR2C
#define ALLOC(var,TYPE,num_objects,routine) \
	var = (TYPE *) calloc((size_t)num_objects,sizeof(TYPE)); \
	if (var == NULL){\
		puts("Error in <"#routine">: Allocation of "#var" failed.");\
		puts("Exiting.");\
		exit(EXIT_FAILURE);\
	}
#define cudaALLOC(var,TYPE,num_objects,routine) \
	cudaMalloc((void**)&var, sizeof(TYPE)*num_objects);  \
	if (var == NULL){\
		puts("Error in GPU <"#routine">: Allocation of "#var" failed.");\
		puts("Exiting.");\
		exit(EXIT_FAILURE);\
	}

typedef cufftReal real;			/* Define type 'real', precision defined in <fftw.h> */
typedef cufftComplex complexy;   /* Define type complex using type real */
cufftHandle pfft,pinv;

extern int rand();			/* extern : permet de dÃ©clarer une fonction dÃ©finie ailleurs */
double arand();
void definitions(real *,real *,real *,real *,real *,real *);
void condition_initiale(int choix);
void dK_x(complexy *,complexy *);
void dK_y(complexy *,complexy *);
void lap();
void integration();
void read_parameters(char *file);
void premiere_fois();
void tfi(complexy *, real *);
void pmoyen(char *filein);
void mean_wavenumber(char *filein);
void write_data(char *filein);
/*void detecteur();*/

real *c,*Psi;
real *cG,*PsiG; 
real *kx,*ky,*k2,*k2t;
real *kxG,*kyG,*k2G,*k2tG;
real *dtk2,*dt2k2;
real *dtk2G,*dt2k2G;
complexy *cc,*PP,*ccold,*JJold;
complexy *ccG,*PPG,*ccoldG,*JJoldG;

int    option,nt2,nt3,alea;
int    compteur = 0;
int    compteurb = 0;

int    back,front;

double tprofil,tconc,tld,total_time,integration_time;
double taille,dt2,R,dt,eps,width;
/***********************************************************************/


__global__ void definitions1(real* Kx, double interm, int CELX)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < CELX/2+1)
		Kx[idx] = idx*interm;	
}


__global__ void definitions2(real* Ky, double interm)
{
	int idy = blockIdx.x * blockDim.x + threadIdx.x;
	Ky[idy] = (((idy+CELY/2-1)%CELY)-CELY/2+1)*interm;	
}

__global__ void definitions3(int idint, real* K2, real* K2T, real* Dtk2,real* Dt2k2, real* Kx, real* Ky, double eps,double dt, double dt2, int CELX)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int id = idx+idint;
	if(idx < CELX/2+1)
	{
		K2[id] = eps*Ky[idy]*Ky[idy]+Kx[idx]*Kx[idx];
		K2T[id] = Ky[idy]*Ky[idy]+Kx[idx]*Kx[idx];
		Dtk2[id] = exp(-dt*K2[id]);
		Dt2k2[id] = dt2*K2[id];
	}
}


__global__ void initialize_zero(real* Psi)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	Psi[id] = 0;

}	

__global__ void dk_xloop(int idint, complexy* dv, complexy *x, int CELX)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < CELX/2 + 1)
	{
		int id =  idx + idint;
		dv[id].x = -kx[idx]*x[id].y;
		dv[id].y = kx[idx]*x[id].x;
	}

}

__global__ void dk_yloop(int idint, complexy* a, complexy *b, int idy, int CELX)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < CELX/2 + 1)
	{
		int id = idx + idint;
		dv[id].x = -ky[idy]*x[id].y;
		dv[id].y = ky[idy]*x[id].x;
	}

}


__global__ void tfiloop(int x)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	x[id] = x[id]/taille;
}


__global__ void premeirefois1(real *Jold, real *N, real *cx, real *cy, real *Psix, real *Psiy)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	Jold[id] = Psiy[id]*cx[id]-Psix[id]*cy[id];
	N[id] = Psix[id]*cx[id]+Psiy[id]*cy[id];
}


__global__ void premierefois2(complexy *cc,complexy *ccold,complexy *JJold, double dt, real *k2, int CELY, int CELX)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	if(id < CELY*(CELX/2+1))
	{
		ccold[id].x = cc[id].x;
		ccold[id].y = cc[id].y;
		cc[id].x = -dt*JJold[id].x+(1.-k2[id]*dt)*cc[id].x;
		cc[id].y = -dt*JJold[id].y+(1.-k2[id]*dt)*cc[id].y;		
	}
}
__global void premierefois3(complexy *PP, double R, complexy *NN, complexy *ccy, real *k2t, int CELX, int CELY)
{
	if(id == 0)
	{
		PP[id].x = 0;
		PP[id].y = 0;
	}
	else if(id < CELY * (CELX/2+1))
	{
		PP[id].x = R*(NN[id].x+ccy[id].x)/k2t[id];
		PP[id].y = R*(NN[id].y+ccy[id].y)/k2t[id];
	}
}

__global__ void integration1(real *J, real *N, real *Psiy, real *Psix,real *cx, real *cy)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	J[id] = Psiy[id]*cx[id]-Psix[id]*cy[id];
	N[id] = Psix[id]*cx[id]+Psiy[id]*cy[id];

}

__global__ void integration2(complexy *ccnew,complexy *cc,double dt, complexy *JJ, complexy *JJold, real *dtk2, int CELX, int CELY)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id < CELY*(CELX/2+1))
	{
		ccnew[id].x = cc[id].x-dt*(1.5*JJ[id].x-0.5*JJold[id].x);
		ccnew[id].x = ccnew[id].x*dtk2[id];
		ccnew[id].y = cc[id].y-dt*(1.5*JJ[id].y-0.5*JJold[id].y);
		ccnew[id].y = ccnew[id].y*dtk2[id];
	}
}

__global__ void integration3(complexy *PP, complexy *NN, complexy *ccy, real *k2t, double R, int CELX, int CELY)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id == 0)
	{
		PP[id].x = 0.;
		PP[id].y = 0.;
	}
	else if(id < CELY*(CELX/2+1))
	{
		PP[id].x = R*(NN[id].x+ccy[id].x)/k2t[id];
		PP[id].y = R*(NN[id].y+ccy[id].y)/k2t[id];
	}
}


__global__ void integration4(real *Jnew, real *Psix, real *Psiy, real *cx, real *cy)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	Jnew[id] = Psiy[id]*cx[id]-Psix[id]*cy[id];

}

__global__ void integration5(complexy *cc, complexy *ccold, double dt2, complexy *JJnew, complexy *JJ, real *dt2k2, complexy *cc)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id < CELY*(CELX/2+1))
	{
		ccold[id].x = cc[id].x;
		ccold[id].y = cc[id].y;
		cc[id].x = cc[id].x-dt2*(JJnew[id].x+JJ[id].x)-dt2k2[id]*(ccnew[id].x+cc[id].x);
		cc[id].y = cc[id].y-dt2*(JJnew[id].y+JJ[id].y)-dt2k2[id]*(ccnew[id].y+cc[id].y);
		JJold[id].x = JJ[id].x;
		JJold[id].y = JJ[id].y;
	}
}



int main(int argc,char **argv)
{
	int seed;
        int i,idx,idy,id,idint,npas;
	int size[2],nligne,opt;
	int *sizeG;
	//cudaALLOC(sizeG,int,2,main);

	double interm;
	char sim_name[60];

	strcpy(sim_name,argv[2]);
								/* strcpy(dest,source) copie la chaine source dans dest							 */
								/* NB: dest doit etre assez grand pour contenir source + un caractere nul de fin */
								/*     donc on a: sim_name = "rout "											 */

	/*******  DEFINITIONS  **********************************/
		/*cudaMalloc((void**)&cc, sizeof(cufftComplex)*CELX*CELY);
		if(cc==NULL)
			printf("Gotcha\n");*/
        cudaALLOC(c,real,CELX*CELY,main);

        cudaALLOC(Psi,real,CELX*CELY,main);
        cudaALLOC(kx,real,CELX/2+1,main);
        cudaALLOC(ky,real,CELY,main);
        cudaALLOC(k2,real,CELY*(CELX/2+1),main);
        cudaALLOC(k2t,real,CELY*(CELX/2+1),main);
        cudaALLOC(dtk2,real,CELY*(CELX/2+1),main);
        cudaALLOC(dt2k2,real,CELY*(CELX/2+1),main);

        cudaALLOC(cc,complexy,CELY*(CELX/2+1),main);
        cudaALLOC(PP,complexy,CELY*(CELX/2+1),main);
        cudaALLOC(ccold,complexy,CELY*(CELX/2+1),main);
        cudaALLOC(JJold,complexy,CELY*(CELX/2+1),main);

        //cudaALLOC(cG,real,CELX*CELY,main);
        //cudaALLOC(PsiG,real,CELX*CELY,main);
        //cudaALLOC(kxG,real,CELX/2+1,main);
        //cudaALLOC(kyG,real,CELY,main);
        //cudaALLOC(k2G,real,CELY*(CELX/2+1),main);
        //cudaALLOC(k2tG,real,CELY*(CELX/2+1),main);
        //cudaALLOC(dtk2G,real,CELY*(CELX/2+1),main);
        //cudaALLOC(dt2k2G,real,CELY*(CELX/2+1),main);
 
        //cudaALLOC(ccG,complexy,CELY*(CELX/2+1),main);
        //cudaALLOC(PPG,complexy,CELY*(CELX/2+1),main);
        //cudaALLOC(ccoldG,complexy,CELY*(CELX/2+1),main);
        //cudaALLOC(JJoldG,complexy,CELY*(CELX/2+1),main);

	size[0]=CELY;
	size[1]=CELX;
	cufftHandle *ptr;
    ptr = &pfft;
    cufftHandle *ptr_inv;
    ptr_inv = &pinv;
    

	cufftPlan2d(&pfft, size[0], size[1], CUFFT_R2C);
	cufftPlan2d(ptr_inv, size[0], size[1], CUFFT_C2R);
	
	
	taille = (double)CELX*CELY;
	read_parameters(argv[1]);	// argv[1] a Ã©tÃ© remplacÃ© par "condinit" 

	npas = integration_time/dt+1;
	nligne = npas/(integration_time/tld);
	front = 10.;
	back = 0.;

/*	seed = 1000;
	srand(seed);*/
	/* initialize random number generator */

	write_data(sim_name);
	definitions(kx,ky,k2,k2t,dtk2,dt2k2);
	printf("****EVERYTHING FINE TILL LINE 134****\n");
	condition_initiale(option);
	/*for(i=1;i<CELX*CELY;i++)
	{
		printf("haha = %f\n",c[i]);
	}*/
	printf("ARAND = %f\n",arand());
	pmoyen(sim_name);
	mean_wavenumber(sim_name);
	premiere_fois();

	for (i = 1; i <= npas; i++){
		printf("Hi\n");
		  total_time = total_time+dt;
		  integration();

	  if(((i+1)%nligne)==0){
	    if(front-back<0){
	    i = npas+1;
	    printf("front=%f back=%f\n",front,back);
	    printf("Les deux fronts se rejoignent a t=%f\n",total_time);
	    }
	    
	    tfi(cc,c);
	    tfi(PP,Psi);

	    pmoyen(sim_name);
	    mean_wavenumber(sim_name);
	   
		tf(pfft,c,cc);

		tf(pfft,Psi,PP);		
	  }
	}
// 	printf("Temps final de la simulation = %f\n",total_time);
// 	printf("------------------------------------------------------------\n\n");

	cudaFree(c);
	cudaFree(ccold);
	cudaFree(JJold);
	cudaFree(cc);
	cudaFree(Psi);
	cudaFree(PP);
	cudaFree(ky);
	cudaFree(kx);
	cudaFree(k2);
	cudaFree(k2t);
	cudaFree(dtk2);
	cudaFree(dt2k2);
	cudaFree(cG);
	cudaFree(ccoldG);
	cudaFree(JJoldG);
	cudaFree(ccG);
	cudaFree(PsiG);
	cudaFree(PPG);
	cudaFree(kyG);
	cudaFree(kxG);
	cudaFree(k2G);
	cudaFree(k2tG);
	cudaFree(dtk2G);
	cudaFree(dt2k2G);
	cudaFree(pfft);
	cudaFree(pinv);
	return 0;
}
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
/*void detecteur()
  {	int  idx,idy,id,idint;
  real *cmx;
  double ddx;

  ddx = (double)Lx/CELX;
  for (idx=0; idx<CELX ; ++idx) cmx[idx] = 0.;
  for (idy = 0; idy < CELY; ++idy){
  idint = idy*CELX;
  for (idx=0 ; idx<CELX ; ++idx){
  id = idint+idx;
  cmx[idx] = cmx[idx]+c[id];
  }
  }

  for(idx=0;idx<CELX;idx++)
  cmx[idx]=cmx[idx]/CELY;


  }*/
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
void pmoyen(char *filein) // filein = rout
{
	int  idx,idy,id,idint;
	double interm,moyenne1,moyenne2,pos,tmp;
	double beginning,area,M1,M2,M3,Mtot,temp,sfing;
	real *cmx,*cmxG;
	double ddx,ddy;
	int d1,d2,c99;
	int longueur=0;
	char file_name[60];
	FILE *fp;
	real *ccc;
	ALLOC(ccc,real,CELX*CELY,pmoyen);
	ALLOC(cmx,real,CELX,pmoyen);
	//cudaALLOC(cmxG,real,CELX,pmoyen);
	cudaMemcpy(ccc,c,sizeof(real)*CELX*CELY,cudaMemcpyDeviceToHost);

	while (filein[longueur] != '\0') longueur += 1;
	longueur += 2;
	M1 = 0.;
	M2 = 0.;
	M3 = 0.;
	Mtot = 0.;
	printf("longueur=%d\n",longueur);
	/*-------------------  Mean profile versus x   ---------------------*/

	ddx = (double)Lx/CELX;
	ddy = (double)Ly/CELY;
	printf("ddx = %f\n",ddx);
	printf("ddy = %f\n",ddy);

	for (idx=0; idx<CELX ; ++idx) cmx[idx] = 0.;


	for (idy = 0; idy < CELY; ++idy){
		idint = idy*CELX;
		for (idx=0 ; idx<CELX ; ++idx){
			id = idint+idx;
			cmx[idx] = cmx[idx]+ccc[id];
		}
	}
	/*----------------   Mixing length ------------------------*/
	moyenne2 = cmx[0]/CELY;
	for(idx=0;idx<CELX;idx++)
		cmx[idx]=cmx[idx]/CELY;

	for(idx=0;idx<CELX;idx++){
		moyenne1 = moyenne2;
		moyenne2 = cmx[idx];
		if(moyenne2>0.01000 && moyenne1<0.01000) back=idx;
		if(moyenne2<0.01000 && moyenne1>0.01000) front=idx;
	}

	/*------   First moment (mean position of the gravity center) -----------*/

	for(idx=0;idx<CELX-1;idx+=2){
		M1 += 2*cmx[idx]*idx;
		Mtot += 2*cmx[idx];
		M1 += 4*cmx[idx+1]*(idx+1);
		Mtot += 4*cmx[idx+1];
	}

	Mtot = Mtot*ddx/3.;
	M1 = M1*ddx*ddx/Mtot/3.;
	printf("Mtot = %f\n",Mtot);


	/*----------------   Second and third moments -------------*/
	for(idx=0;idx<CELX-1;idx+=2){
		interm = idx*ddx-M1;
		temp = interm*interm;
		M2 += 2*cmx[idx]*temp;
		M3 += 2*cmx[idx]*temp*interm;
		interm = (idx+1)*ddx-M1;
		temp = interm*interm;
		M2 += 4*cmx[idx+1]*temp;
		M3 += 4*cmx[idx+1]*temp*interm;
	}
	M2 = M2*ddx/Mtot/3.;
	M3 = M3*ddx/Mtot/3.;
	sfing = M2-(width*width)/12.-2.*total_time;
	sfing = sqrt(sfing);
	printf("M3 = %f\n",M3);
	/*----------------Writing in files -------------*/


	strcpy(file_name,filein);
	strcat(file_name,"L");
	fp = fopen(file_name,"a");
	fprintf(fp,"%f %f\n",total_time,(front-back)*ddx);
	fclose(fp);



	strcpy(file_name,filein);
	strcat(file_name,"M1");
	fp = fopen(file_name,"a");
	fprintf(fp,"%f %f %f\n",total_time,M1,Mtot);
	fclose(fp);

	strcpy(file_name,filein);
	strcat(file_name,"M2");
	fp = fopen(file_name,"a");
	fprintf(fp,"%f %f\n",total_time,M2);
	fclose(fp);

	strcpy(file_name,filein);
	strcat(file_name,"M3");
	fp = fopen(file_name,"a");
	fprintf(fp,"%f %f\n",total_time,M3);
	fclose(fp);

	strcpy(file_name,filein);
	strcat(file_name,"SF");
	fp = fopen(file_name,"a");
	fprintf(fp,"%f %f\n",total_time,sfing);
	fclose(fp);

	if((compteur%nt2)==0){           /* Printing every tprofil times */


		/*------ -------------- Mean profile along X ------------*/

		strcpy(file_name,filein);
		strcat(file_name,"X");
		fp = fopen(file_name,"a");
		for(idx=0;idx<CELX;idx++){
			interm = idx*ddx;
			moyenne2 = cmx[idx];
			fprintf(fp,"%f %f %d\n",interm,moyenne2,idx);
		}
		fprintf(fp,"\n");
		fclose(fp);

	}
	/*-------------------- Concentration matrix------------*/
	if((compteur%nt3)==0){
		d1 = fmod(compteurb,10);
		d2 = fmod((compteurb-d1)/10,10);
		strcpy(file_name,filein);
		strcat(file_name,"_c");

		file_name[longueur]=d2+'0';
		file_name[longueur+1]=d1+'0';
		file_name[longueur+2]='\0';
		fp = fopen(file_name,"w");
		for (idx=0 ; idx<CELX ; ++idx){
			for (idy = 0; idy < CELY; ++idy){
				idint = idy*CELX;
				id = idx+idint;
				fprintf(fp,"%f ",ccc[id]);
			}
			fprintf(fp,"\n");

		}
		fclose(fp);


		/*-------------------- Stream function matrix   ------------*/

		/*	  strcpy(file_name,filein);
			  strcat(file_name,"_p");
			  file_name[longueur]=d2+'0';
			  file_name[longueur+1]=d1+'0';
			  file_name[longueur+2]='\0';

			  fp = fopen(file_name,"w");
			  for (idx=0 ; idx<CELX ; ++idx){
			  for (idy = 0; idy < CELY; ++idy){
			  idint = idy*CELX;
			  id = idx+idint;
			  fprintf(fp,"%f ",Psi[id]);
			  }
			  fprintf(fp,"\n");
			  }
			  fclose(fp);*/

		compteurb += 1;
	}

	compteur += 1;
	free(cmx);

}
/**************************************************************************/
void mean_wavenumber(char *filein)
{
	int  i,j,idx,idy,id,idint,imax;
	char file_name[60];
	real *ccc;
	ALLOC(ccc,real,CELX*CELY,pmoyen);
	cudaMemcpy(ccc,c,sizeof(real)*CELX*CELY,cudaMemcpyDeviceToHost);

	real *temp,*cmy,*tempG,*cmyG;
	double *P;
	complexy *ttemp,*ttempG;
	double ddy,Pmax,kmean,Ptot;
	int nmode;
	FILE *fp,*gp,*pp,*lp;

	ALLOC(temp,real,CELY*CELX,mean_wavenumber);
	ALLOC(cmy,real,CELY,mean_wavenumber);
	ALLOC(ttemp,complexy,CELY*(CELX/2+1),mean_wavenumber);
	cudaALLOC(tempG,real,CELY*CELX,mean_wavenumber);
	cudaALLOC(cmyG,real,CELY,mean_wavenumber);
	cudaALLOC(ttempG,complexy,CELY*(CELX/2+1),mean_wavenumber);
	ALLOC(P,double,CELY/2+1,mean_wavenumber);

 	strcpy(file_name,filein);
	strcat(file_name,"K");
	fp = fopen(file_name,"a");
 	strcpy(file_name,filein);
	strcat(file_name,"Kmax");
	pp = fopen(file_name,"a");
 	strcpy(file_name,filein);

	lp = fopen(file_name,"a");

	/********* Initialization ***************/
	nmode = 0;
	kmean = 0.;
	Pmax = 0.;
	Ptot = 0.;

	for (idy = 0; idy < CELY; ++idy){
	  idint = idy*CELX;
	  cmy[idy] = 0.;
	  for (idx=0 ; idx<CELX ; ++idx){
	    id = idx+idint;
	    temp[id] = 0.;
	  }
	}

	/********* Mean Profile along Y ***************/

   	for (idy = 0; idy < CELY; ++idy){
		idint = idy*CELX;
		for (idx=0 ; idx<CELX ; ++idx){
			id = idx+idint;
			cmy[idy] = cmy[idy]+ccc[id];
			}
		}

   	for (idy = 0; idy < CELY; ++idy){
		idint = idy*CELX;
		cmy[idy] = cmy[idy]/CELX;
		for (idx=0 ; idx<CELX ; ++idx){
			id = idx+idint;
			temp[id] = cmy[idy];
			}
		}

	/********** Power spectrum ************/
	/*for(i=0;i<CELX*CELY;i++)
	{
		printf("%f\n",temp[i]);
		if(temp[i]!=0)
			break;
	}*/
		//TEMP HAS NON ZERO VALUES
	cudaMemcpy(tempG, temp,CELY*CELX*sizeof(real), cudaMemcpyHostToDevice);
	tf(pfft,tempG,ttempG);
	cudaMemcpy(ttemp, ttempG, CELY*(CELX/2+1)*sizeof(complexy), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
/*	for(i=0;i<CELY*(CELX/2+1);i++)
	{

		printf("%f\n",temp[i]);
		if(ttemp[i].x!=0)
			printf("%f\n",ttemp[i].x);
			
	}
*/
	//printf("i=%d\n",i);
	//printf("CELY*(CELX/2+1)=%d\n",CELY*(CELX/2+1));

	

	for(idy = 1; idy <= CELY/2; ++idy){
		idint = idy*(CELX/2+1);
		for (idx=0 ; idx<CELX/2+1 ; idx++){
			id = idx+idint;
			P[idy] += ttemp[id].x*ttemp[id].x+ttemp[id].y*ttemp[id].y;
			  }
		P[idy] = sqrt(P[idy]);
		Ptot += P[idy];
		if(P[idy]>Pmax){
		  imax = idy;
		  Pmax = P[idy];
		}	
		nmode +=1;
	}
	/*for(i=0;i<CELY/2+1;i++)
		printf("P=%lf\n",P[i]);*/
	printf("ptot=%lf\n",Ptot);
	printf("pmax=%lf\n",Pmax);

	for(idy = 1; idy <= CELY/2; ++idy){
	  kmean += P[idy]*idy/Ptot;
	}
	printf("kmean=%f\n",kmean);

	fprintf(fp,"%f %f\n",total_time,kmean);
	fprintf(pp,"%f %d %f\n",total_time,imax,Pmax);

	fclose(fp);
	fclose(pp);

	/*************** Trace du profil moyen ********/
	if((compteur%nt2)==0){
	  strcpy(file_name,filein);
	  strcat(file_name,"Y");
	  gp = fopen(file_name,"a");
	  for (idy = 0; idy < CELY; ++idy){
		ddy = idy*Ly/CELY;
		fprintf(gp,"%f %f\n",ddy,cmy[idy]);
	  }
	fprintf(gp,"\n");
	fclose(gp);
	}

	free(temp);
	free(ttemp);
	free(cmy);
	free(P);
	cudaFree(tempG);
	cudaFree(ttempG);
	cudaFree(cmyG);
}
/**************************************************************************/
void tfi(complexy *xx,real *x)
{
	// int id;
	// complexy *xxG;
	// real *xG;

	// cudaALLOC(xxG,complexy,CELY*(CELX/2+1),tfi);
	// cudaALLOC(xG,real,CELX*CELY,tfi);

	// cudaMemcpy(xxG, xx, CELY*(CELX/2+1)*sizeof(complexy), cudaMemcpyHostToDevice);	
	tfinv(pinv,xx,x);
	// cudaMemcpy(x, xG,CELY*CELX*sizeof(real), cudaMemcpyDeviceToHost);

	// cudaDeviceSynchronize();
	tfiloop<<<1024,1024>>>(x);
	// for(id=0;id<CELY*CELX;id++) x[id] = x[id]/taille;
	// cudaFree(xxG);
	// cudaFree(xG);
}
/**************************************************************************/
void premiere_fois()
{
	int id,idy;
	real *Jold,*N,*JoldG, *NG;
	real *cx,*cy;
	real *Psix,*Psiy;
	complexy *deriv,*ccy,*NN, *NNG;

	cudaALLOC(Jold,real,CELX*CELY,premiere_fois);
	cudaALLOC(N,real,CELX*CELY,premiere_fois);
	cudaALLOC(JoldG,real,CELX*CELY,premiere_fois);
	cudaALLOC(NG,real,CELX*CELY,premiere_fois);
	cudaALLOC(cx,real,CELX*CELY,premiere_fois);
	cudaALLOC(cy,real,CELX*CELY,premiere_fois);
	cudaALLOC(Psix,real,CELX*CELY,premiere_fois);
	cudaALLOC(Psiy,real,CELX*CELY,premiere_fois);
	cudaALLOC(deriv,complexy,CELY*(CELX/2+1),premiere_fois);
	cudaALLOC(NN,complexy,CELY*(CELX/2+1),premiere_fois);

	tf(pfft,c,cc);
	// cudaMemcpy(cc, ccG, CELY*(CELX/2+1)*sizeof(complexy), cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();

	// cudaMemcpy(PsiG, Psi,CELY*CELX*sizeof(real), cudaMemcpyHostToDevice);
	tf(pfft,Psi,PP);
	// cudaMemcpy(PP, PPG, CELY*(CELX/2+1)*sizeof(complexy), cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();

	total_time = total_time+dt;

	/*********** DERIVEES DANS L'ESPACE DE FOURIER  **************/
	dK_y(cc,deriv);
	tfi(deriv,cy);
	dK_x(cccc,deriv);
	tfi(deriv,cx);
	dK_y(PP,deriv);
	tfi(deriv,Psiy);
	dK_x(PP,deriv);
	tfi(deriv,Psix);
	cudaFree(deriv);

	/*********** NON LINEARITES DANS L'ESPACE REEL  **************/
	tfi(cc,c);

	premierefois1<<<1024,1024>>>(Jold, N, cx, cy, Psix, Psiy);

	// for (id = 0; id < CELX*CELY; ++id)
	// {
	// 	Jold[id] = Psiy[id]*cx[id]-Psix[id]*cy[id];
	// 	N[id] = Psix[id]*cx[id]+Psiy[id]*cy[id];
	// }

	tf(pfft,c,cc);
	
	tf(pfft,Jold,JJold);


	tf(pfft,N,NN);
	cudaFree(Jold);
	cudaFree(N);

	/*********** TERMES LINEAIRES DANS L'ESPACE DE FOURIER  *******/

	premierefois2<<<513,1024>>>(cc,ccold,JJold,dt,k2, CELY, CELX);

	cudaALLOC(ccy,complexy,CELY*(CELX/2+1),integration);
	dK_y(cc,ccy);

	premierefois3<<<513,1024>>>(PP, R, NN, ccy, k2t, CELX, CELY)

	
	cudaFree(ccy);
	cudaFree(cx);
	cudaFree(cy);
	cudaFree(Psix);
	cudaFree(Psiy);
	cudaFree(NN);
	cudaFree(JoldG);
	cudaFree(NG);
	cudaFree(JJoldG);
	cudaFree(NNG);

}
/**************************************************************************/
void integration()
{
	int id,idy;
	real *J,*N,*JG,*NG;
	real *Psix,*Psiy;
	real *cx,*cy,*Jnew,*cnew,*cnewG, *JnewG;
	complexy *deriv,*ccy,*JJnew,*JJnewG,*ccnew,*JJ,*NN,*JJG,*NNG,*ccnewG;

	cudaALLOC(J,real,CELX*CELY,integration);
	cudaALLOC(N,real,CELX*CELY,integration);
	cudaALLOC(cx,real,CELX*CELY,integration);
	cudaALLOC(cy,real,CELX*CELY,integration);
	cudaALLOC(Psix,real,CELX*CELY,integration);
	cudaALLOC(Psiy,real,CELX*CELY,integration);
	cudaALLOC(Jnew,real,CELX*CELY,integration);
	cudaALLOC(cnew,real,CELX*CELY,integration);
	//cudaALLOC(cnewG,real,CELX*CELY,integration);
	//cudaALLOC(JG,real,CELX*CELY,integration);
	//cudaALLOC(NG,real,CELX*CELY,integration);
	//cudaALLOC(JnewG,real,CELX*CELY,integration);

	cudaALLOC(deriv,complexy,CELY*(CELX/2+1),integration);
	cudaALLOC(ccnew,complexy,CELY*(CELX/2+1),integration);
	cudaALLOC(JJ,complexy,CELY*(CELX/2+1),integration);
	cudaALLOC(ccy,complexy,CELY*(CELX/2+1),integration);
	cudaALLOC(JJnew,complexy,CELY*(CELX/2+1),integration);
	cudaALLOC(NN,complexy,CELY*(CELX/2+1),integration);
	//cudaALLOC(JJG,complexy,CELY*(CELX/2+1),integration);
	//cudaALLOC(NNG,complexy,CELY*(CELX/2+1),integration);
	//cudaALLOC(ccnewG,complexy,CELY*(CELX/2+1),integration);
	//cudaALLOC(JJnewG,complexy,CELY*(CELX/2+1),integration);

	dK_y(cc,deriv);
	tfi(deriv,cy);
	dK_x(cc,deriv);
	tfi(deriv,cx);
	dK_y(PP,deriv);
	tfi(deriv,Psiy);
	dK_x(PP,deriv);
	tfi(deriv,Psix);

	tfi(cc,c);
	
	integration1<<<1024,1024>>>(J,N,Psix,Psix,cx,cy);
	/*======  Espace de Fourier ===============================*/

	tf(pfft,J,JJ);
	tf(pfft,N,NN);
	cudaFree(J);
	cudaFree(N);
	tf(pfft,c,cc);

	integration2<<<513,1024>>>(ccnew,cc,dt,JJ,JJold,dtk2,CELX,CELY);

	dK_y(ccnew,ccy);  /* necessaire car c a change en cours de route */

	integration3<<<513,1024>>>(PP, NN, ccy, k2t, R, CELX, CELY);
	tfi(ccy,cy);
	cudaFree(ccy);
	dK_x(ccnew,deriv);
	tfi(deriv,cx);
	dK_y(PP,deriv);
	tfi(deriv,Psiy);
	dK_x(PP,deriv);
	tfi(deriv,Psix);

	tfi(ccnew,cnew);

	integration4<<<1024,1024>>> (Jnew, Psix, Psiy, cx, cy);

	tf(pfft,Jnew,JJnew);
	tf(pfft,cnew,ccnew);

	integration5<<<513,1024>>> (cc,ccold,dt2,JJnew,JJ,dt2k2,ccnew);

	cudaFree(JJnew);
	cudaFree(ccnew);
	cudaFree(deriv);
	cudaFree(JJ);
	cudaFree(cx);
	cudaFree(cy);
	cudaFree(Psix);
	cudaFree(Psiy);
	cudaFree(cnew);
	cudaFree(Jnew);
	cudaFree(NN);
	cudaFree(JG);
	cudaFree(NG);
	cudaFree(JnewG);
	cudaFree(cnewG);
	cudaFree(JJG);
	cudaFree(NNG);
	cudaFree(JJnewG);
	cudaFree(ccnewG);
	
}
/*=========================================================================*/
void condition_initiale(int choix)
{
	real* ccc;
	ALLOC(ccc, real, CELX*CELY, condition_initiale);
	
	printf("****CONDITION INITIALE STARTS\n****");
	int i,idx,idy,id,idint,finalea;
	double pos,ddx,z1,z2,kgauche,vgauche,interm,zut1, fixc;
	int debut,fin;
	ddx = (double)Lx/CELX;
	if(choix == 1){
		printf("Condition initiale = stepfunction + bruit sur c=1/2 \n");

		initialize_zero<<<1024,1024>>>(Psi);

		if(R<0.)debut = 1*(Lx-width)/5./ddx;
		else debut=4*(Lx-width)/5./ddx;
		/*	  fin = (Lx+width)/2./ddx;*/
		fin = debut + width/ddx;

		printf("debut=%f fin=%f largeur=%f milieu=%f\n",debut*ddx,fin*ddx,(fin-debut)*ddx,(fin+debut)*ddx/2.);

		finalea = alea*CELY;
		printf("finalea=%d\n",finalea);
		for(idy=0; idy<finalea;++idy)
		{
			zut1 = arand();
			printf("zut = %f\n",zut1);
		} 

		//printf("%f\n",zut1);
		for (idy = 0; idy < CELY; ++idy)
		{
			idint = idy*CELX;
			
			for (idx=0 ; idx<debut ; ++idx)
			{
				id = idx+idint;
				ccc[id] = 0.;
			}
			
			for (idx=debut ; idx<debut+1 ; ++idx)
			{
				id = idx+idint;
				fixc = 1./2.*(1.+0.001*arand());
				ccc[id]=fixc;
			}
			
			for (idx=debut+1 ; idx<fin ; ++idx)
			{
				id = idx+idint;
				ccc[id] = 1.;
			}
		
			for (idx=fin ; idx<fin+1 ; ++idx)
			{
				id = idx+idint;
				ccc[id] = 1-fixc;
			}

			for (idx=fin+1 ; idx<CELX; ++idx)
			{
				id = idx+idint;
				ccc[id] = 0.;
			}
}
//printf("c=%lf\n",c[id]);

}
	cudaMemcpy(c, ccc, sizeof(real)*CELX*CELY, cudaMemcpyHostToDevice);
}

/*=========================================================================*/
void definitions(real Kx[],real Ky[],real K2[],real K2T[],real Dtk2[],real Dt2k2[])
{
	printf("***INSIDE DEFINITION FUNCTION****\n");
	int  idx,idy,idint,id;
	double interm;

	interm = M_2PI/Lx;
	definitions1<<<5,820>>>(Kx, interm, CELX);

	interm = M_2PI/Ly;
	definitions2<<<1,128>>>(Ky,interm);

	for(idy=0;idy<CELY;idy+=1){
		idint = idy*(CELX/2+1);
		definitions3<<<5,820>>>(idint, K2, K2T, Dtk2, Dt2k2, Kx, Ky, eps, dt, dt2, CELX);
	}

	printf("***DEFINITIONS ENDS****\n");
}
/*=========================================================================*/
void dK_x(complexy x[],complexy dv[])
{
	int idx,idy,id,idint;
	for (idy = 0; idy < CELY; ++idy){
		idint = idy*(CELX/2+1);
		dk_xloop<<<5,820>>>(idint,dv,x, CELX);
		
		// for(idx=0;idx<CELX/2+1;idx+=1){
		// 	id = idx+idint;
		// 	dv[id].x = -kx[idx]*x[id].y;
		// 	dv[id].y = kx[idx]*x[id].x;
		// }
	}
}
/*=========================================================================*/
void dK_y(complexy x[],complexy dv[])
{
	int idx,idy,id,idint;

	for (idy = 0; idy < CELY; ++idy){
		idint = idy*(CELX/2+1);
		dk_yloop<<<5,820>>>(idint,dv,x, idy, CELX);
		
		// for(idx=0;idx<CELX/2+1;idx+=1){
		// 	id = idx+idint;
		// 	dv[id].x = -ky[idy]*x[id].y;
		// 	dv[id].y = ky[idy]*x[id].x;
		// }
	}
}
/*=========================================================================*/
void lap(complexy x[],complexy dx[])
{
	int  id;

	for (id = 0; id < CELY*(CELX/2+1); ++id){
		dx[id].x = -k2t[id]*x[id].x;
		dx[id].y = -k2t[id]*x[id].y;
	}
}
/*=========================================================================*/
void read_parameters(char *file)
{
	FILE *rp;

	printf("\n----------------------------------------------------------\n\n");
#if PURE_DF == 1
	printf(" Pure viscous fingering - no chemical reaction\n");
#endif
	printf("\n----------------------------------------------------------\n\n\n");

	rp = fopen(file,"r");
	fscanf(rp,"option = %d\n",&option);
	fscanf(rp,"alea = %d\n",&alea);
	fscanf(rp,"R = %lf\n",&R);
	fscanf(rp,"eps = %lf\n",&eps);
	fscanf(rp,"dt = %lf\n",&dt);
	fscanf(rp,"integration_time = %lf\n",&integration_time);
	fscanf(rp,"total_time = %lf\n",&total_time);
	fscanf(rp,"tld = %lf\n",&tld);
	fscanf(rp,"tprofil = %lf\n",&tprofil);
	fscanf(rp,"tconc = %lf\n",&tconc);
	fscanf(rp,"width = %lf\n",&width);

	printf("****INPUT FILE PARAMETERS READING STARTS****\n");

	printf("option = %d\n",option);
	printf("alea = %d\n",alea);
	printf("R = %lf\n",R);
	printf("eps = %lf\n",eps);
	printf("dt = %lf\n",dt);
	printf("int_time = %lf\n",integration_time);
	printf("total_time = %lf\n",total_time);
	printf("tld = %lf\n",tld);
	printf("tprofil = %lf\n",tprofil);
	printf("tconc = %lf\n",tconc);
	printf("width = %lf\n",width);

	printf("****INPUT FILE PARAMETERS READING ENDS****\n");


	dt2 = dt/2.;
	nt2 = tprofil/tld;
	nt3 = tconc/tld;
	printf(" R=%f  eps=%lf  l=%f  \n\n",R,eps,width);		/*printf(" R=%f  eps=%lf  l=%f  Pe=%d\n\n",R,eps,width,(double)Lx/CELX);*/
	printf(" Pe=%d dx=%10.8f dy=%10.8f A=%d dt=%f\n",Ly,(double)Lx/CELX,(double)Ly/CELY,CELX/CELY,dt);
	printf("   (CELX=%d CELY=%d Lx=%d Ly=%d)\n",CELX,CELY,Lx,Ly);
	printf("    nmodes=%d\n",CELX*CELY);
	printf("    integration_time=%f\n",integration_time);
	printf("    sigma_init = %lf\n",width*width/12.);
	fclose(rp);
}
/**************************************************************************/
double arand()
{
	double z;
	z = (double) (1.0*rand()/RAND_MAX);
	return(z);
}
/*=========================================================================*/
void write_data(char *filein)
{
	char file_name[60];
	int nnpas,nnligne,nn;
	FILE *fp;

	strcpy(file_name,filein);
	strcat(file_name,"D");
	printf("*** OPEN ROUTD TO CHECK IF EVERYTHING IS PRINTED CORRECTLY****\n");
	fp = fopen(file_name,"w");

#if PURE_DF == 1
	fprintf(fp,"\n-----------------------------------------------\n\n");
	fprintf(fp,"  Pure viscous fingering - no chemical reaction\n\n");
#endif

	fprintf(fp,"  Donnees de la simulation %s\n\n",filein);
	fprintf(fp,"-----------------------------------------------\n");

	fprintf(fp,"Pe = %d          A=%f\n\n",Ly,(double)Lx/Ly);

	fprintf(fp,"CELX = %d\n",CELX);
	fprintf(fp,"CELY = %d\n",CELY);
	fprintf(fp,"Lx = %d\n",Lx);
	fprintf(fp,"Ly = %d\n",Ly);
	fprintf(fp,"dx = %4.2f\n",(double)Lx/CELX);
	fprintf(fp,"dy = %4.2f\n\n",(double)Ly/CELY);

	fprintf(fp,"CI = %d\n",option);
	fprintf(fp,"alea = %d\n",alea);
	fprintf(fp,"R = %lf\n",R);
	fprintf(fp,"eps = %lf\n",eps);
	fprintf(fp,"width = %lf\n",width);
	fprintf(fp,"dt = %lf\n",dt);
	fprintf(fp,"tld = %lf\n",tld);
	fprintf(fp,"tprofil = %lf\n",tprofil);
	fprintf(fp,"tconc = %lf\n",tconc);
	fprintf(fp,"total_time = %lf\n",total_time);
	fprintf(fp,"integ_time = %lf\n",integration_time);

	nnpas = integration_time/dt+1;
	nnligne = integration_time/tld;
	nn = integration_time/tprofil;
	fprintf(fp,"\nnpas=%d    nligne=%d  nprofil=%d\n",nnpas,nnligne,nn);

	nn = integration_time/tconc;
	fprintf(fp,"\nnconc=%d\n",nn);
	fprintf(fp,"-----------------------------------------------\n\n");

	fclose(fp);
}
