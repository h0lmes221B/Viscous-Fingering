/* Programme of Viscous fingering */
/* applicable to the pure viscous fingering.							*/

/*To compile, type:
gcc -O3 -o a.out ***.c -lm -lfftw -lrfftw

To run the programme type "./a.out data rout"

This produce a family of output files "rout_Suffix" which
contain the various informations on the run.

For cluster 

icc -O3 -tpp7 -xP -rcd -o a.out ****.c -I./myfftw/include ./myfftw/lib/librfftw.a ./myfftw/lib/libfftw.a

programme for negtive R  with streamfunction and vorticity, Interfacial lengths, positive and negative lengths*/



#include <stdio.h>
#include </home/holmes/Downloads/fftw-2.1.5/threads/fftw_threads.h>
#include </home/holmes/Downloads/fftw-2.1.5/threads/rfftw_threads.h>
#include <math.h>
#include <string.h>
#define num_threads 8
#define PURE_DF  1

#define CELX 1024 		/* Number of grids or meshs according to x */
#define CELY 128       /* Number of grids or meshs according to y */
#define Lx   4096     /* length of the H-S cell */
#define Ly   512		/* width of the H-S cell equal to the Pe */

#define M_PI_2	1.57079632679489661923
#define M_2PI	6.2831853071795864769

#define tf rfftwnd_threads_one_real_to_complex
#define tfinv rfftwnd_threads_one_complex_to_real
#define ALLOC(var,TYPE,num_objects,routine) \
	var = (TYPE *) calloc((size_t)num_objects,sizeof(TYPE)); \
	if (var == NULL){\
		puts("Error in <"#routine">: Allocation of "#var" failed.");\
		puts("Exiting.");\
		exit(EXIT_FAILURE);\
}

typedef fftw_real real;			/* Define type 'real', precision defined in <fftw.h> */
typedef fftw_complex complex;   /* Define type complex using type real */
fftwnd_plan pfft,pinv;

extern int rand();			/* extern : permet de déclarer une fonction définie ailleurs */
double arand();
void definitions();
void condition_initiale(int choix);
void dK_x();
void dK_y();
void lap();
void integration();
void read_parameters(char *file);
void premiere_fois();
void tfi();
void pmoyen(char *filein);
void mean_wavenumber(char *filein);
void write_data(char *filein);
/*void detecteur();*/

real *c,*Psi,*omg;
real *c1x,*c1y;
real *kx,*ky,*k2,*k2t;
real *dtk2,*dt2k2;
complex *cc,*PP,*ccold,*JJold,*OMG,*dervx,*dervy;

int    option,nt2,nt3,alea;
int    compteur = 0;
int    compteurb = 0;
int    back,front;
int    debut,fin; 

double tprofil,tconc,tld,total_time,integration_time;
double taille,dt2,R,dt,eps,width;
/***********************************************************************/
int main(int argc,char **argv)
{
	int fftw_threads_init(void);
	int seed;
    int i,idx,idy,id,idint,npas;
	int size[2],nligne,opt;
	double interm;
	char sim_name[60];

	strcpy(sim_name,argv[2]);      												
								/* strcpy(dest,source) copie la chaine source dans dest							 */
								/* NB: dest doit etre assez grand pour contenir source + un caractere nul de fin */
								/*     donc on a: sim_name = "rout "											 */
	
	/*******  DEFINITIONS  **********************************/

        ALLOC(c,real,CELX*CELY,main);
        ALLOC(c1x,real,CELX*CELY,main);
        ALLOC(c1y,real,CELX*CELY,main);
        ALLOC(Psi,real,CELX*CELY,main);
	ALLOC(omg,real,CELX*CELY,main);
        ALLOC(kx,real,CELX/2+1,main);
        ALLOC(ky,real,CELY,main);
        ALLOC(k2,real,CELY*(CELX/2+1),main);
        ALLOC(k2t,real,CELY*(CELX/2+1),main);
        ALLOC(dtk2,real,CELY*(CELX/2+1),main);
        ALLOC(dt2k2,real,CELY*(CELX/2+1),main);
 
        ALLOC(cc,complex,CELY*(CELX/2+1),main);
        ALLOC(PP,complex,CELY*(CELX/2+1),main);
	ALLOC(OMG,complex,CELY*(CELX/2+1),main);
        ALLOC(ccold,complex,CELY*(CELX/2+1),main);
        ALLOC(JJold,complex,CELY*(CELX/2+1),main);
        ALLOC(dervx,complex,CELY*(CELX/2+1),main);
        ALLOC(dervy,complex,CELY*(CELX/2+1),main);



	size[0]=CELY;
	size[1]=CELX;
       
	pfft = rfftwnd_create_plan(2,size,FFTW_REAL_TO_COMPLEX,FFTW_ESTIMATE);
	pinv = rfftwnd_create_plan(2,size,FFTW_COMPLEX_TO_REAL,FFTW_ESTIMATE); 
	taille = (double)CELX*CELY;
	read_parameters(argv[1]);	/* argv[1] a été remplacé par "condinit" */

	npas = integration_time/dt;
	nligne = npas/(integration_time/tld);
	front = 10.;
	back = 0.;

/*	seed = 1000; 
	srand(seed);*/
	/* initialize random number generator */

	write_data(sim_name);
	definitions(kx,ky,k2,k2t,dtk2,dt2k2);
	condition_initiale(option);
	
        tf(num_threads,pfft,c,cc);
        dK_y(cc,dervy);
        tfi(dervy,c1y);
        dK_x(cc,dervx);
        tfi(dervx,c1x);

        pmoyen(sim_name);
	mean_wavenumber(sim_name);
	premiere_fois();

	for (i = 1; i <= npas; i++){
	  total_time = total_time+dt;
	  integration();

	  if(((i+1)%nligne)==0){	  
	    if(front-back<0){
	    i = npas+1;
	    printf("front=%f back=%f\n",front,back);
	    printf("Les deux fronts se rejoignent a t=%f\n",total_time);
	    }


             dK_y(cc,dervy);
             tfi(dervy,c1y);
             dK_x(cc,dervx);
             tfi(dervx,c1x);      

	     tfi(cc,c);
	     tfi(PP,Psi);
	     tfi(OMG,omg);
	    pmoyen(sim_name);
	    mean_wavenumber(sim_name);
	    tf(num_threads,pfft,c,cc);
	    tf(num_threads,pfft,Psi,PP);
	  }
	}
	printf("Temps final de la simulation = %f\n",total_time);
	printf("------------------------------------------------------------\n\n");

        free(dervy);
        free(dervx);
        free(c1x);
        free(c1y);

	free(c);
	free(ccold);
	free(JJold);
	free(cc);
	free(Psi);
	free(PP);
	free(ky);
	free(kx);
	free(k2);
	free(k2t);
	free(dtk2);
	free(dt2k2);
	free(pfft);
	free(pinv);
	free(omg);
	free(OMG);
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
void pmoyen(char *filein)
{	int  idx,idy,id,idint;
	double interm,moyenne1,moyenne2,pos,tmp;
	double beginning,area,M1,M2,M3,Mtot,temp,sfing,IL, cmax;
	real *cmx;
	double ddx,ddy;
	int d1,d2,c99;
	int longueur=0;
	char file_name[60];
	FILE *fp; 

        ALLOC(cmx,real,CELX,pmoyen);

	while (filein[longueur] != '\0') longueur += 1;
	longueur += 2;
	M1 = 0.;
	M2 = 0.;
	M3 = 0.;
	Mtot = 0.;
        IL=0.;
        cmax=0;
/*-------------------  Mean profile versus x   ---------------------*/      

	ddx = (double)Lx/CELX;
	ddy = (double)Ly/CELY;
	for (idx=0; idx<CELX ; ++idx) cmx[idx] = 0.;


   	for (idy = 0; idy < CELY; ++idy){
		idint = idy*CELX;
		for (idx=0 ; idx<CELX ; ++idx){
			id = idint+idx;
			cmx[idx] = cmx[idx]+c[id];
			}
		}
/*----------------   Mixing length ------------------------*/
	moyenne2 = cmx[0]/CELY;	
	for(idx=0;idx<CELX;idx++){
	  cmx[idx]=cmx[idx]/CELY;
          if(cmx[idx]>cmax) cmax=cmx[idx];
          }

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
	


/* -----------------Interfacial length -----------*/



 for (idy = 0; idy < CELY; ++idy){
                idint = idy*CELX;
                for (idx=0 ; idx<CELX ; ++idx){
                        id = idint+idx;
                        IL = IL+sqrt(c1x[id]*c1x[id]+c1y[id]*c1y[id]);
                        }
                }
IL=IL*ddx*ddy;






/*----------------Writing in files -------------*/
        strcpy(file_name,filein);
        strcat(file_name,"L1");
        fp = fopen(file_name,"a");
        fprintf(fp,"%f %f\n",total_time,IL);
        fclose(fp);


 	strcpy(file_name,filein);
	strcat(file_name,"L");
	fp = fopen(file_name,"a");
	fprintf(fp,"%f %f\n",total_time,(front-back)*ddx);
	fclose(fp);


        strcpy(file_name,filein);
        strcat(file_name,"LP");
        fp = fopen(file_name,"a");
        fprintf(fp,"%f %f\n",total_time,(front-fin)*ddx);
        fclose(fp);



        strcpy(file_name,filein);
        strcat(file_name,"LM");
        fp = fopen(file_name,"a");
        fprintf(fp,"%f %f\n",total_time,(debut-back)*ddx);
        fclose(fp);

        strcpy(file_name,filein);
        strcat(file_name,"Cmax");
        fp = fopen(file_name,"a");
        fprintf(fp,"%f %f\n",total_time,cmax);
        fclose(fp);

 	strcpy(file_name,filein);
	strcat(file_name,"M1");
	fp = fopen(file_name,"a");
	fprintf(fp,"%f %f %f\n",total_time,M1,Mtot);
	fclose(fp); 	strcpy(file_name,filein);
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
		fprintf(fp,"%f ",c[id]);
		}
		fprintf(fp,"\n");

	    }
	  fclose(fp);


/*-------------------- Stream function matrix   ------------*/
         strcpy(file_name,filein);
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
		fclose(fp);

	
/*-------------------- Vorticity matrix   ------------*/
  strcpy(file_name,filein);
	  strcat(file_name,"_o");
	   file_name[longueur]=d2+'0';   
	  file_name[longueur+1]=d1+'0';
	  file_name[longueur+2]='\0';

	  fp = fopen(file_name,"w");
	  for (idx=0 ; idx<CELX ; ++idx){
	    for (idy = 0; idy < CELY; ++idy){
		idint = idy*CELX;
		id = idx+idint;
		fprintf(fp,"%f ",omg[id]);
		}
	        fprintf(fp,"\n");
		}
		fclose(fp);

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
	
	real *temp,*cmy;
	double *P;
	complex *ttemp;
	double ddy,Pmax,kmean,Ptot;
	int nmode;
	FILE *fp,*gp,*pp,*lp; 

	ALLOC(temp,real,CELY*CELX,mean_wavenumber);
	ALLOC(cmy,real,CELY,mean_wavenumber);
	ALLOC(ttemp,complex,CELY*(CELX/2+1),mean_wavenumber);
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
			cmy[idy] = cmy[idy]+c[id];
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

	tf(num_threads,pfft,temp,ttemp);

	for(idy = 1; idy <= CELY/2; ++idy){
		idint = idy*(CELX/2+1);
		for (idx=0 ; idx<CELX/2+1 ; idx++){
			id = idx+idint;
			P[idy] += ttemp[id].re*ttemp[id].re+ttemp[id].im*ttemp[id].im;
			  }
		P[idy] = sqrt(P[idy]);
		Ptot += P[idy];
		if(P[idy]>Pmax){
		  imax = idy;
		  Pmax = P[idy];
		}
		nmode +=1;
	}

	for(idy = 1; idy <= CELY/2; ++idy){
	  kmean += P[idy]*idy/Ptot;
	}

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
}
/**************************************************************************/
void tfi(complex *xx,real *x)
{
  int id;

  tfinv(num_threads,pinv,xx,x);
  for(id=0;id<CELY*CELX;id++) x[id] = x[id]/taille;
}
/**************************************************************************/
void premiere_fois()
{
	int id,idy;
	real *Jold,*N;
	real *cx,*cy;
	real *Psix,*Psiy;
	complex *deriv,*ccy,*NN;

        ALLOC(Jold,real,CELX*CELY,premiere_fois);
        ALLOC(N,real,CELX*CELY,premiere_fois);
        ALLOC(cx,real,CELX*CELY,premiere_fois);
        ALLOC(cy,real,CELX*CELY,premiere_fois);
        ALLOC(Psix,real,CELX*CELY,premiere_fois);
        ALLOC(Psiy,real,CELX*CELY,premiere_fois);
        ALLOC(deriv,complex,CELY*(CELX/2+1),premiere_fois);
        ALLOC(NN,complex,CELY*(CELX/2+1),premiere_fois);

	tf(num_threads,pfft,c,cc);
	tf(num_threads,pfft,Psi,PP);

	total_time = total_time+dt;

	/*********** DERIVEES DANS L'ESPACE DE FOURIER  **************/
	dK_y(cc,deriv);
	tfi(deriv,cy);
	dK_x(cc,deriv);
	tfi(deriv,cx);
	dK_y(PP,deriv);
	tfi(deriv,Psiy);
	dK_x(PP,deriv);
	tfi(deriv,Psix);
	free(deriv);

	/*********** NON LINEARITES DANS L'ESPACE REEL  **************/

	tfi(cc,c);
	for (id = 0; id < CELX*CELY; ++id){
	  Jold[id] = Psiy[id]*cx[id]-Psix[id]*cy[id];
	  N[id] = Psix[id]*cx[id]+Psiy[id]*cy[id];
		}
	tf(num_threads,pfft,c,cc);
	tf(num_threads,pfft,Jold,JJold);
	tf(num_threads,pfft,N,NN);

	free(Jold);
	free(N);

	/*********** TERMES LINEAIRES DANS L'ESPACE DE FOURIER  *******/
	for (id = 0; id < CELY*(CELX/2+1); ++id){
	  ccold[id].re = cc[id].re;	  
	  ccold[id].im = cc[id].im;	  
	  cc[id].re = -dt*JJold[id].re+(1.-k2[id]*dt)*cc[id].re;
	  cc[id].im = -dt*JJold[id].im+(1.-k2[id]*dt)*cc[id].im;
	}
        ALLOC(ccy,complex,CELY*(CELX/2+1),integration);	
	dK_y(cc,ccy);

	for (id = 1; id < CELY*(CELX/2+1); ++id){
		OMG[id].re= -R*(NN[id].re+ccy[id].re);
		OMG[id].im=-R*(NN[id].im+ccy[id].im);
		PP[id].re = -OMG[id].re/k2t[id];
		PP[id].im = -OMG[id].im/k2t[id];
	}
       	PP[0].re = 0.;
	PP[0].im = 0.;
        OMG[0].re=0.;
        OMG[0].im=0.;
	
	free(ccy);
	free(cx);
	free(cy);
	free(Psix);
	free(Psiy);
	free(NN);
       
}
/**************************************************************************/
void integration()
{
	int id,idy;
	real *J,*N;
	real *Psix,*Psiy;
	real *cx,*cy,*Jnew,*cnew;
	complex *deriv,*ccy,*JJnew,*ccnew,*JJ,*NN;

        ALLOC(J,real,CELX*CELY,integration); 
        ALLOC(N,real,CELX*CELY,integration); 
        ALLOC(cx,real,CELX*CELY,integration);
        ALLOC(cy,real,CELX*CELY,integration);
        ALLOC(Psix,real,CELX*CELY,integration);
        ALLOC(Psiy,real,CELX*CELY,integration);
        ALLOC(Jnew,real,CELX*CELY,integration);
        ALLOC(cnew,real,CELX*CELY,integration);

        ALLOC(deriv,complex,CELY*(CELX/2+1),integration);
        ALLOC(ccnew,complex,CELY*(CELX/2+1),integration);
        ALLOC(JJ,complex,CELY*(CELX/2+1),integration);
        ALLOC(ccy,complex,CELY*(CELX/2+1),integration);
        ALLOC(JJnew,complex,CELY*(CELX/2+1),integration);	
        ALLOC(NN,complex,CELY*(CELX/2+1),integration);	

	dK_y(cc,deriv);
	tfi(deriv,cy);
	dK_x(cc,deriv);
	tfi(deriv,cx);
	dK_y(PP,deriv);
	tfi(deriv,Psiy);
	dK_x(PP,deriv);
	tfi(deriv,Psix);

	tfi(cc,c);

/*=====   Derivees dans l'espace reel =====================*/

	for (id = 0; id < CELX*CELY; ++id){
		J[id] = Psiy[id]*cx[id]-Psix[id]*cy[id];
		N[id] = Psix[id]*cx[id]+Psiy[id]*cy[id];
		}
/*======  Espace de Fourier ===============================*/

	tf(num_threads,pfft,J,JJ);
	tf(num_threads,pfft,N,NN);
	free(J);
	free(N);

	tf(num_threads,pfft,c,cc);

	for (id = 0; id < CELY*(CELX/2+1); ++id){
	  ccnew[id].re = cc[id].re-dt*(1.5*JJ[id].re-0.5*JJold[id].re);
	  ccnew[id].re = ccnew[id].re*dtk2[id];
	  ccnew[id].im = cc[id].im-dt*(1.5*JJ[id].im-0.5*JJold[id].im);
	  ccnew[id].im = ccnew[id].im*dtk2[id];
	}


	dK_y(ccnew,ccy);  /* necessaire car c a change en cours de route */

	for (id = 1; id < CELY*(CELX/2+1); ++id){
         	
		OMG[id].re= -R*(NN[id].re+ccy[id].re);
		OMG[id].im=-R*(NN[id].im+ccy[id].im);
		PP[id].re = -OMG[id].re/k2t[id];
		PP[id].im = -OMG[id].im/k2t[id];
	}
       	PP[0].re = 0.;
		PP[0].im = 0.;
		OMG[0].re=0.;
		OMG[0].im=0.;
	    
       
		
	tfi(ccy,cy);
	free(ccy);

	dK_x(ccnew,deriv);
	tfi(deriv,cx);
	dK_y(PP,deriv);
	tfi(deriv,Psiy);
	dK_x(PP,deriv);
	tfi(deriv,Psix);

	tfi(ccnew,cnew);

	for (id = 0; id < CELX*CELY; ++id)	/* termes non lineaires */
		Jnew[id] = Psiy[id]*cx[id]-Psix[id]*cy[id];

	tf(num_threads,pfft,Jnew,JJnew);
	tf(num_threads,pfft,cnew,ccnew);

	for (id = 0; id < CELY*(CELX/2+1); ++id){
	 	ccold[id].re = cc[id].re;
	        ccold[id].im = cc[id].im;
		cc[id].re = cc[id].re-dt2*(JJnew[id].re+JJ[id].re)
				-dt2k2[id]*(ccnew[id].re+cc[id].re);
		cc[id].im = cc[id].im-dt2*(JJnew[id].im+JJ[id].im)
				-dt2k2[id]*(ccnew[id].im+cc[id].im);
		JJold[id].re = JJ[id].re;
		JJold[id].im = JJ[id].im;
		}
	free(JJnew);
	free(ccnew);
	free(deriv);
	free(JJ);
	free(cx);
	free(cy);
	free(Psix);
	free(Psiy);
	free(cnew);
	free(Jnew);
	free(NN);
}
/*=========================================================================*/
void condition_initiale(int choix)
{
        int i,idx,idy,id,idint,finalea;
	double pos,ddx,z1,z2,kgauche,vgauche,interm,zut1, fixc;
/*	int debut,fin; */ 
	ddx = (double)Lx/CELX;
	if(choix == 1){
	  printf("Condition initiale = stepfunction + bruit sur c=1/2 \n");

	  for (id=0;id<CELX*CELY;++id) Psi[id] = 0.;

	  if(R<0.)debut = 1*(Lx-width)/5./ddx;
	  else debut=4*(Lx-width)/5./ddx;
	  /*	  fin = (Lx+width)/2./ddx;*/
	  fin = debut + width/ddx;

	  printf("debut=%f fin=%f largeur=%f milieu=%f\n",debut*ddx,fin*ddx,(fin-debut)*ddx,(fin+debut)*ddx/2.);

	  finalea = alea*CELY;
	  for(idy=0; idy<finalea;++idy) zut1 = arand();

	printf("%f\n",zut1);  
	  for (idy = 0; idy < CELY; ++idy){
	    idint = idy*CELX;
	    for (idx=0 ; idx<debut ; ++idx){
	      id = idx+idint;
	      c[id] = 0.;}
	    for (idx=debut ; idx<debut+1 ; ++idx){
	      id = idx+idint;
	      /*	      c[id] = 1./2.*(1.+0.001*arand());}*/     
	      fixc = 1./2.*(1.+0.001*arand());
              c[id]=fixc;      
	      }
	   
            for (idx=debut+1 ; idx<fin ; ++idx){
	      id = idx+idint;
	      c[id] = 1.;}
	    for (idx=fin ; idx<fin+1 ; ++idx){
	      id = idx+idint;
	      /*	      c[id] = 1./2.*(1.+0.001*arand());}*/
	      c[id] = 1-fixc;}
            	   
            for (idx=fin+1 ; idx<CELX; ++idx){
	      id = idx+idint;
	      c[id] = 0.;}
	  }
       
	}
}

/*=========================================================================*/
void definitions(real Kx[],real Ky[],real K2[],real K2T[],real Dtk2[],real Dt2k2[])
{
	int  idx,idy,idint,id;
	double interm;

	interm = M_2PI/Lx;
	for(idx=0;idx<CELX/2+1;idx+=1){
		Kx[idx] = idx*interm;
		}

	interm = M_2PI/Ly;
	for(idy=0;idy<CELY;idy+=1){
		Ky[idy] = (((idy+CELY/2-1)%CELY)-CELY/2+1)*interm;
	}
	for(idy=0;idy<CELY;idy+=1){
		idint = idy*(CELX/2+1);
		for(idx=0;idx<CELX/2+1;idx+=1){
		  id = idx+idint;
		  K2[id] = eps*Ky[idy]*Ky[idy]+Kx[idx]*Kx[idx];
		  K2T[id] = Ky[idy]*Ky[idy]+Kx[idx]*Kx[idx];
		  Dtk2[id] = exp(-dt*K2[id]);
		  Dt2k2[id] = dt2*K2[id];
		}}
}
/*=========================================================================*/
void dK_x(complex x[],complex dv[])
{
	int idx,idy,id,idint;
   	for (idy = 0; idy < CELY; ++idy){
		idint = idy*(CELX/2+1);
		for(idx=0;idx<CELX/2+1;idx+=1){
			id = idx+idint;
			dv[id].re = -kx[idx]*x[id].im;
			dv[id].im = kx[idx]*x[id].re;
			}}
}
/*=========================================================================*/
void dK_y(complex x[],complex dv[])
{
	int idx,idy,id,idint;

   	for (idy = 0; idy < CELY; ++idy){
		idint = idy*(CELX/2+1);
		for(idx=0;idx<CELX/2+1;idx+=1){
			id = idx+idint;
			dv[id].re = -ky[idy]*x[id].im;
			dv[id].im = ky[idy]*x[id].re;
			}}
}
/*=========================================================================*/
void lap(complex x[],complex dx[])
{
	int  id;

   	for (id = 0; id < CELY*(CELX/2+1); ++id){
		dx[id].re = -k2t[id]*x[id].re;
		dx[id].im = -k2t[id]*x[id].im;
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
  nnligne = nnpas/(integration_time/tld);
  nn = integration_time/tprofil;
  fprintf(fp,"\nnpas=%d    nligne=%d  nprofil=%d\n",nnpas,nnligne,nn);

  nn = integration_time/tconc;
  fprintf(fp,"\nnconc=%d\n",nn);
  fprintf(fp,"-----------------------------------------------\n\n");

  fclose(fp);
}
