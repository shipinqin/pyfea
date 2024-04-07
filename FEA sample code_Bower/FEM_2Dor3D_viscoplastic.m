function FEM_2Dor3D_viscoplastic
%
%          Example 2D and 3D Linear elastic FEM code
%            Currently coded to run either plane strain or plane stress (2DOF) or general 3D but
%            could easily be modified for axisymmetry too.
%
%        Variables read from input file;
%        nprops              No. material parameters
%        materialprops(i)    List of material parameters
%        ncoord              No. spatial coords (2 for 2D, 3 for 3D)
%        ndof                No. degrees of freedom per node (2 for 2D, 3 for 3D)
%                            (here ndof=ncoord, but the program allows them to be different
%                            to allow extension to plate & beam elements with C^1 continuity)
%        nnode               No. nodes
%        coords(i,j)         ith coord of jth node, for i=1..ncoord; j=1..nnode
%        nelem               No. elements
%        maxnodes            Max no. nodes on any one element (used for array dimensioning)
%        nelnodes(i)         No. nodes on the ith element
%        elident(i)          An integer identifier for the ith element.  Not used
%                            in this code but could be used to switch on reduced integration,
%                            etc.
%        connect(i,j)        List of nodes on the jth element
%        nfix                Total no. prescribed displacements
%        fixnodes(i,j)       List of prescribed displacements at nodes
%                            fixnodes(1,j) Node number
%                            fixnodes(2,j) Displacement component number (1, 2 or 3)
%                            fixnodes(3,j) Value of the displacement
%        ndload              Total no. element faces subjected to tractions
%        dloads(i,j)         List of element tractions
%                            dloads(1,j) Element number
%                            dloads(2,j) face number
%                            dloads(3,j), dloads(4,j), dloads(5,j) Components of traction
%                            (assumed uniform) 
%
%  To run the program you first need to set up an input file, as described in 
%  the lecture notes.  Then change the fopen command below to point to the file.
%  Also change the fopen command in the post-processing step (near the bottom of the
%  program) to point to a suitable output file.  Then execute the file in
%  the usual fashion (e.g. hit the green arrow at the top of the MATLAB
%  editor window)
% 
%
% ==================== Read data from the input file ===========================
%
%
% YOU NEED TO CHANGE THE PATH & FILE NAME TO POINT TO YOUR INPUT FILE
%
infile=fopen('viscoplastic_quad4.txt','r');
outfile=fopen('FEM_results.txt','w');

[nprops,materialprops,ncoord,ndof,nnode,coords,nelem,maxnodes,connect,nelnodes,elident,nfix,fixnodes,ndload,dloads] = read_input_file(infile);

fclose(infile);

% Plot the initial mesh as a check
close all
figure
plotmesh(coords,ncoord,nnode,connect,nelem,elident,nelnodes,'g');

%
%============================ MAIN FEM ANALYSIS PROCEDURE ========================
%
%   w           Nodal displacements.  Let w_i^a be ith displacement component
%               at jth node.  Then dofs contain (w_1^1, w_2^1, w_1^2, w_2^2....) for 2D
%               and (w_1^1, w_2^1, w_3^1, w_1^2, w_2^2, w_3^2....) for 3D
%   dw          Correction to nodal displacements.  Let w_i^a be ith displacement component
%               at jth node.  Then dofs contain (w_1^1, w_2^1, w_1^2, w_2^2....) for 2D
%               and (w_1^1, w_2^1, w_3^1, w_1^2, w_2^2, w_3^2....) for 3D
%   K           Global stiffness matrix.  Stored as [K_1111 K_1112  K_1121  K_1122...
%                                                    K_1211 K_1212  K_1221  K_1222...
%                                                    K_2111 K_2112  K_2121  K_2122...]
%               for 2D problem and similarly for 3D problem
%   F           Force vector.  Currently only includes contribution from tractions
%               acting on element faces (i.e. body forces are neglected)
%   R           Volume contribution to residual
%   b           RHS of equation system
%
  w = zeros(nnode*ndof,1);
  u = w;
%
% Set up arrays to store the element state
%
  nintp = numberofintegrationpoints(ncoord,nelnodes(1),elident(1));
  stress = zeros(3,3,nintp,nelem);
  eplas = zeros(nintp,nelem);
%
%  Here we specify how the Newton Raphson iteration should run
%  Load is applied in nsteps increments;
%  tol is the tolerance used in checking Newton-Raphson convergence
%  maxit is the max no. Newton-Raphson iterations
%  relax is the relaxation factor (Set to 1 unless big convergence problems)
%
  nsteps = 10;
  tol = 0.0001;
  maxit = 30;
  relax = 1.;
  dt = 2./nsteps;      


  forcevdisp = zeros(2,nsteps+1);
  forcevdisp(1,1) = 0;
  forcevdisp(2,1) = 0;

  for step = 1 : nsteps

     loadfactor = step/nsteps;

     err1 = 1.;
     nit = 0;
   
    fprintf(1,'\n Step %f Load %f\n',step,loadfactor);

     while ((err1>tol) && (nit<maxit))          % Newton Raphson loop
         nit = nit + 1;

         K = globalstiffness(dt,ncoord,ndof,nnode,coords, ...
                 nelem,maxnodes,elident,nelnodes,connect,materialprops,stress,eplas,w);
         F = globaltraction(ncoord,ndof,nnode,ndload,coords, ...
                      nelnodes,elident,connect,dloads,w);
         R = globalresidual(dt,ncoord,ndof,nnode,coords, ...
                    nelem,maxnodes,elident,nelnodes, ...
                          connect,materialprops,stress,eplas,w);
         
         b = loadfactor*F - R;
         
%          Fix constrained nodes.
         for n = 1 : nfix
           rw = ndof*(fixnodes(1,n)-1) + fixnodes(2,n);
           for cl = 1 : ndof*nnode
             K(rw,cl) = 0;
           end
           K(rw,rw) = 1.;
           b(rw) = loadfactor*fixnodes(3,n)-w(rw)-u(rw);
         end
%
%        Solve for the correction
%
         dw = K\b;
%
%        Check convergence
%

         w = w + relax*dw;
         wnorm = dot(w,w);
         err1 = dot(dw,dw);
         err2 = dot(b,b);
         err1 = sqrt(err1/wnorm);
         err2 = sqrt(err2)/(ndof*nnode);
         fprintf(1,'Iteration number %d Correction %f Residual %f tolerance %f\n',nit,err1,err2,tol);
     end
% Update the stress and accumulated plastic strain
     [stress,eplas] = update_state(dt,ncoord,ndof,nnode,coords,nelem,maxnodes, ...
         elident,nelnodes,connect,materialprops,stress,eplas,w);
% Update the total displacecment
     u = u + w;
% 
% Print the current solution
%
    fprintf(outfile,'\n\n Step %f Load %f\n',step,loadfactor);

  print_results(outfile, ...
     nprops,materialprops,ncoord,ndof,nnode,coords, ...
     nelem,maxnodes,connect,nelnodes,elident,stress,eplas, ...
     nfix,fixnodes,ndload,dloads,u);
%
 %      Store traction and displacement for plotting later
%
    forcevdisp(2,step+1) = loadfactor*dloads(3,1);
    forcevdisp(1,step+1) = u(3);
  end 

  plot(forcevdisp(1,:),forcevdisp(2,:),'r','LineWidth',3);
  xlabel({'Displacement'},'FontSize',16);
  ylabel({'Force'},'FontSize',16);

  
%
%================================= POST-PROCESSING =================================
%
% Create a plot of the deformed mesh
%

  defcoords = zeros(ndof,nnode); 
  scalefactor = 1.0;
  for i = 1:nnode
    for j = 1:ndof
       defcoords(j,i) = coords(j,i) + scalefactor*w(ndof*(i-1)+j); 
    end
  end

 figure
 plotmesh(coords,ncoord,nnode,connect,nelem,elident,nelnodes,'g');
 hold on
 plotmesh(defcoords,ncoord,nnode,connect,nelem,elident,nelnodes,'r');  
  

 
fclose(outfile); 
end 

%======================= deplas ===================
%
%  Computes plastic strain increment de_e given strain increment
%
function e = deplas(dt,stress,eplas,dstrain,materialprops)

%  Youngs modulus and Poissons ratio
   E = materialprops(1);
   nu = materialprops(2);
   Y = materialprops(3);
   e0 = materialprops(4);
   n = materialprops(5);
   edot0 = materialprops(6);
   m = materialprops(7);
%
%  S is the deviatoric stress predictor
%
   S = zeros(3,3);
   de = zeros(3,3);
   dl = [ [1,0,0];[0,1,0];[0,0,1] ];  

   devol = trace(dstrain);
   p = trace(stress);
   
   sequiv = 0.;
   for i = 1 : 3
     for j = 1 : 3
        de(i,j) = dstrain(i,j) - dl(i,j)*devol/3;
        S(i,j) = stress(i,j) - dl(i,j)*p/3 + E/(1+nu)*de(i,j);
        sequiv = sequiv + S(i,j)*S(i,j);
     end
   end
   sequiv = sqrt(1.5*sequiv);

   e = 10^(-15);
   err = Y;
   tol = 10^(-06)*Y;
   if (sequiv*edot0 == 0)
     e = 0.;
   else
     while (err>tol)
       c = (1+(eplas+e)/e0)^(1/n)*(e/(dt*edot0))^(1/m);
       f = sequiv/Y - 1.5*e*E/(Y*(1+nu))- c;
       dfde = -1.5*E/(Y*(1+nu)) - c*(1/(n*(eplas+e+e0)) + 1/(m*e));
       enew = e - f/dfde;
       if (enew<0)
%        e must be >0, so if new approx to e <0 the solution
%        must lie between current approx to e and zero.
         e = e/10;
       else
         e = enew;
       end
       err = abs(f);
     end
   end

end
%================= Material Stiffness ==================================
%
%    Computes material stiffness tensor C_{ijkl} 
%    Currently coded either for plane strain or general 3D.
%    Note that in this procedure `stress' is the current estimate
%    for stress at the end of the increment S_n+1
%
function C =  materialstiffness(ndof,ncoord,stress,eplas,dep,dstrain,materialprops)
   
%  Bulk modulus, Youngs modulus and Poissons ratio
   K = materialprops(1)/(3*(1-2*materialprops(2)));
   E = materialprops(1);
   nu = materialprops(2);
   e0 = materialprops(4);
   n = materialprops(5);
   m = materialprops(7);
   
%
%  S is the current approx for deviatoric stress at end of increment
%
   S = zeros(3,3);
   de = zeros(3,3);
   dl = [ [1,0,0];[0,1,0];[0,0,1] ]; 

   devol = trace(dstrain);
   p = trace(stress);
   
   se = 0;
   for i = 1 : 3
     for j = 1 : 3
        de(i,j) = dstrain(i,j) - dl(i,j)*devol/3;
        S(i,j) = stress(i,j) - dl(i,j)*p/3;
        se = se + S(i,j)*S(i,j);
     end
   end
   se = sqrt(1.5*se);

   if (se*dep>0)
     beta = 1./(1+1.5*E*dep/((1+nu)*se));
     gamma = beta*( 1.5*E/((1+nu)*se)+(1/(n*(e0+eplas+dep))+1/(m*dep)) );
     factor = 1.5*1.5*E*(dep-1/gamma)/((1+nu)*se^3);
   else
     beta = 1.;   
     factor = 0.;
   end     
   if (ncoord == 2)  
     for i = 1 : 2
       for j = 1 : 2
         for k = 1 : 2
           for l = 1 : 2
             C(i,j,k,l) = beta*E/(1+nu)*((dl(i,k)*dl(j,l)+dl(j,k)*dl(i,l))/2-dl(i,j)*dl(k,l)/3 ...
                                   + factor*S(i,j)*S(k,l)) + K*dl(i,j)*dl(k,l);
           end
         end
       end
     end
        
   elseif (ncoord == 3)

     for i = 1 : 3
       for j = 1 : 3
         for k = 1 : 3
           for l = 1 : 3
             C(i,j,k,l) = beta*E/(1+nu)*((dl(i,k)*dl(j,l)+dl(j,k)*dl(i,l))/2-dl(i,j)*dl(k,l)/3 ...
                                   + factor*S(i,j)*S(k,l)) + K*dl(i,j)*dl(k,l);
            end
          end
        end
      end
    end


end
%================= Material Stress ==================================
%
%   Computes updated stress sigma_{ij}^n+1 given strain increment depsilon_{ij}
%   Also returns a value for increment in accumulated plastic strain dep
%
function S = materialstress(dt,ndof,ncoord,stress,eplas,dep,dstrain,materialprops)

%  Bulk modulus, Youngs modulus and Poissons ratio
   K = materialprops(1)/(3*(1-2*materialprops(2)));
   E = materialprops(1);
   nu = materialprops(2);
%
%  S is the deviatoric stress predictor
%
   S = zeros(3,3);
   de = zeros(3,3);
   dl = [ [1,0,0];[0,1,0];[0,0,1] ];  

   devol = trace(dstrain);
   p = trace(stress);
   
   se = 0.;
   for i = 1 : 3
     for j = 1 : 3
        de(i,j) = dstrain(i,j) - dl(i,j)*devol/3;
        S(i,j) = stress(i,j) - dl(i,j)*p/3 + E/(1+nu)*de(i,j);
        se = se + S(i,j)*S(i,j);
     end
   end
   se = sqrt(1.5*se);

   if (se>0)
     beta = 1 - 1.5*E*dep/((1+nu)*se);
   else
     beta = 1.;
   end      
%  S now stores the full stress tensor       
   for i = 1 : 3
     for j = 1 : 3
         S(i,j) = beta*S(i,j) + (p/3+K*devol)*dl(i,j);
     end
   end
end
%

%====================== No. integration points =============================
%
%   Defines the number of integration points:be used for
%   each element type
%
function n = numberofintegrationpoints(ncoord,nelnodes,elident)
  
   if (ncoord == 1) 
     n = nelnodes;   
   elseif (ncoord == 2) 
     if (nelnodes == 3)
         n = 1;
     end
     if (nelnodes == 6)
         n = 3;
     end
     if (nelnodes == 4)
         n = 4;
     end
     if (nelnodes == 8)
         n = 9;
     end
   elseif (ncoord == 3) 
     if (nelnodes == 4)
         n = 1 ;
     end
     if (nelnodes == 10)
         n = 4;
     end
     if (nelnodes == 8)
         n = 8;
     end
     if (nelnodes == 20)
         n = 27;
     end
   end
end   
%
%====================== INTEGRATION POINTS ==================================
%
%   Defines positions of integration points
%
function xi = integrationpoints(ncoord,nelnodes,npoints,elident)

   xi = zeros(ncoord,npoints);
%
%  1D elements
%
   if (ncoord == 1) 
     if (npoints==1) 
       xi(1,1) = 0.;
     elseif (npoints == 2) 
       xi(1,1) = -0.5773502692;
       xi(1,2) = -xi(1,1);
     elseif (npoints == 3) 
       xi(1,1) = -0.7745966692;
       xi(1,2) = 0.0;
       xi(1,3) = -xi(1,1);
     end
%
%  2D elements
%
   elseif (ncoord == 2) 
%
%    Triangular element
%
     if ( nelnodes == 3 || nelnodes == 6 ) 
       if (npoints == 1) 
         xi(1,1) = 1./3.;
         xi(2,1) = 1./3.;
       elseif (npoints == 3) 
         xi(1,1) = 0.6;
         xi(2,1) = 0.2;
         xi(1,2) = 0.2;
         xi(2,2) = 0.6;
         xi(1,3) = 0.2;
         xi(2,3) = 0.2;
       elseif (npoints == 4) 
         xi(1,1) = 1./3.;
         xi(2,1) = 1./3.;
         xi(1,2) = 0.6;
         xi(2,2) = 0.2;
         xi(1,3) = 0.2;
         xi(2,3) = 0.6;
         xi(1,4) = 0.2;
         xi(2,4) = 0.2;
       end
%
%    Rectangular element
%                  
     elseif ( nelnodes==4 || nelnodes==8 ) 

       if (npoints == 1) 
         xi(1,1) = 0.;
         xi(2,1) = 0.;
       elseif (npoints == 4) 
         xi(1,1) = -0.5773502692;
         xi(2,1) = xi(1,1);
         xi(1,2) = -xi(1,1);
         xi(2,2) = xi(1,1);
         xi(1,3) = xi(1,1);
         xi(2,3) = -xi(1,1);
         xi(1,4) = -xi(1,1);
         xi(2,4) = -xi(1,1);
       elseif (npoints == 9) 
         xi(1,1) = -0.7745966692;
         xi(2,1) = xi(1,1);
         xi(1,2) = 0.0;
         xi(2,2) = xi(1,1);
         xi(1,3) = -xi(1,1);
         xi(2,3) = xi(1,1);
         xi(1,4) = xi(1,1);
         xi(2,4) = 0.0;
         xi(1,5) = 0.0;
         xi(2,5) = 0.0;
         xi(1,6) = -xi(1,1);
         xi(2,6) = 0.0;
         xi(1,7) = xi(1,1);
         xi(2,7) = -xi(1,1);
         xi(1,8) = 0.;
         xi(2,8) = -xi(1,1);
         xi(1,9) = -xi(1,1);
         xi(2,9) = -xi(1,1);
       end
     end
%
%   3D elements
%
   elseif (ncoord == 3) 
%
%  3D elements
%
     if (nelnodes == 4 || nelnodes==10 ) 
       if (npoints == 1) 
         xi(1,1) = 0.25;
         xi(2,1) = 0.25;
         xi(3,1) = 0.25;
       elseif (npoints == 4) 
         xi(1,1) = 0.58541020;
         xi(2,1) = 0.13819660;
         xi(3,1) = xi(2,1);
         xi(1,2) = xi(2,1);
         xi(2,2) = xi(1,1);
         xi(3,2) = xi(2,1);
         xi(1,3) = xi(2,1);
         xi(2,3) = xi(2,1);
         xi(3,3) = xi(1,1);
         xi(1,4) = xi(2,1);
         xi(2,4) = xi(2,1);
         xi(3,4) = xi(2,1);
       end
     elseif ( nelnodes==8 || nelnodes==20 ) 
       if (npoints == 1) 
         xi(1,1) = 0.;
         xi(2,1) = 0.;
         xi(3,1) = 0.;
       elseif (npoints == 8) 
         x1D = [-0.5773502692,0.5773502692];
         for k = 1:2
           for j = 1:2 
             for i = 1:2
               n = 4*(k-1) + 2*(j-1) + i;
               xi(1,n) = x1D(i);
               xi(2,n) = x1D(j);
               xi(3,n) = x1D(k);
             end
           end
         end
       elseif (npoints == 27) 
         x1D = [-0.7745966692,0.,0.7745966692];
         for k = 1:3
           for j = 1:3
             for i = 1:3
               n = 9*(k-1) + 3*(j-1) + i;
               xi(1,n) = x1D(i);
               xi(2,n) = x1D(j);
               xi(3,n) = x1D(k);
             end
           end
         end
       end
     end
   end
end

%
%================= INTEGRATION WEIGHTS ==================================
%
%   Defines integration weights w_i
%
function w = integrationweights(ncoord,nelnodes,npoints,elident)

   w = zeros(npoints,1);

%
%  1D elements
%
   if (ncoord == 1) 
     if (npoints == 1)
       w(1) = 2.;
     elseif (npoints == 2) 
       w = [1.,1.];
     elseif (npoints == 3) 
       w = [0.555555555,0.888888888,0.555555555];
     end
%
%  2D elements
%
   elseif (ncoord == 2) 
%
%    Triangular element
%
     if ( nelnodes == 3 || nelnodes == 6 ) 
       if (npoints == 1) 
         w(1) = 0.5;
       elseif (npoints == 3) 
         w(1) = 1./6.;
         w(2) = 1./6.;
         w(3) = 1./6.;
       elseif (npoints == 4) 
         w = [-27./96.,25./96.,25/96.,25/96.];
       end
%
%    Rectangular element
%                  
     elseif ( nelnodes==4 || nelnodes==8 ) 

       if (npoints == 1) 
         w(1) = 4.;
       elseif (npoints == 4) 
         w = [1.,1.,1.,1.];
       elseif (npoints == 9 ) 
         w1D = [0.555555555,0.888888888,0.55555555555];
         for j = 1:3
           for i = 1:3
             n = 3*(j-1)+i;
             w(n) = w1D(i)*w1D(j);
           end
         end    
       end
     end 

   elseif (ncoord == 3) 
%
%  3D elements
%
     if (nelnodes == 4 || nelnodes==10 ) 
       if (npoints == 1) 
         w(1) = 1./6.;
       elseif (npoints == 4) 
         w = [1./24.,1./24.,1./24.,1./24.];
       end
     elseif ( nelnodes==8 || nelnodes==20 ) 
       if (npoints == 1) 
         w(1) = 8.;
       elseif (npoints == 8) 
         w = [1.,1.,1.,1.,1.,1.,1.,1.];
       elseif (npoints == 27) 
         w1D = [0.555555555,0.888888888,0.55555555555];
         for k = 1:3
           for j = 1:3
             for i = 1:3
               n = 9*(k-1)+3*(j-1)+i;
               w(n) = w1D(i)*w1D(j)*w1D(k);
             end
           end    
         end
       end
     end
   end
end

%
%================= SHAPE FUNCTIONS ==================================
%
%        Calculates shape functions for various element types
%
function N = shapefunctions(nelnodes,ncoord,elident,xi)
 

   N = zeros(nelnodes,1);
%
%  1D elements
%
  if (ncoord == 1) 
    if (nelnodes==2) 
      N(1) = 0.5*(1.+xi(1));
      N(2) = 0.5*(1.-xi(1));
    elseif (nelnodes == 3) 
      N(1) = -0.5*xi(1)*(1.-xi(1));
      N(2) =  0.5*xi(1)*(1.+xi(1));
      N(3) = (1.-xi(1))*(1.+xi(1));
    end
%
%  2D elements
%
   elseif (ncoord == 2) 
%
%    Triangular element
%
     if ( nelnodes == 3 ) 
       N(1) = xi(1);
       N(2) = xi(2);
       N(3) = 1.-xi(1)-xi(2);               
     elseif ( nelnodes == 6 ) 
       xi3 = 1.-xi(1)-xi(2);
       N(1) = (2.*xi(1)-1.)*xi(1);
       N(2) = (2.*xi(2)-1.)*xi(2);
       N(3) = (2.*xi3-1.)*xi3;
       N(4) = 4.*xi(1)*xi(2);
       N(5) = 4.*xi(2)*xi3;
       N(6) = 4.*xi3*xi(1);
%
%    Rectangular element
%                  
     elseif ( nelnodes == 4 ) 
       N(1) = 0.25*(1.-xi(1))*(1.-xi(2));
       N(2) = 0.25*(1.+xi(1))*(1.-xi(2));
       N(3) = 0.25*(1.+xi(1))*(1.+xi(2));
       N(4) = 0.25*(1.-xi(1))*(1.+xi(2));
     elseif (nelnodes == 8) 
       N(1) = -0.25*(1.-xi(1))*(1.-xi(2))*(1.+xi(1)+xi(2));
       N(2) = 0.25*(1.+xi(1))*(1.-xi(2))*(xi(1)-xi(2)-1.);
       N(3) = 0.25*(1.+xi(1))*(1.+xi(2))*(xi(1)+xi(2)-1.);
       N(4) = 0.25*(1.-xi(1))*(1.+xi(2))*(xi(2)-xi(1)-1.);
       N(5) = 0.5*(1.-xi(1)*xi(1))*(1.-xi(2));
       N(6) = 0.5*(1.+xi(1))*(1.-xi(2)*xi(2));
       N(7) = 0.5*(1.-xi(1)*xi(1))*(1.+xi(2));
       N(8) = 0.5*(1.-xi(1))*(1.-xi(2)*xi(2));
     end
%
   elseif (ncoord==3) 

     if (nelnodes == 4) 
       N(1) = xi(1);
       N(2) = xi(2);
       N(3) = xi(3);
       N(4) = 1.-xi(1)-xi(2)-xi(3);
     elseif (nelnodes == 10) 
       xi4 = 1.-xi(1)-xi(2)-xi(3);
       N(1) = (2.*xi(1)-1.)*xi(1);
       N(2) = (2.*xi(2)-1.)*xi(2);
       N(3) = (2.*xi(3)-1.)*xi(3);
       N(4) = (2.*xi4-1.)*xi4;
       N(5) = 4.*xi(1)*xi(2);
       N(6) = 4.*xi(2)*xi(3);
       N(7) = 4.*xi(3)*xi(1);
       N(8) = 4.*xi(1)*xi4;
       N(9) = 4.*xi(2)*xi4;
       N(10) = 4.*xi(3)*xi4;
     elseif (nelnodes == 8) 
       N(1) = (1.-xi(1))*(1.-xi(2))*(1.-xi(3))/8.;
       N(2) = (1.+xi(1))*(1.-xi(2))*(1.-xi(3))/8.;
       N(3) = (1.+xi(1))*(1.+xi(2))*(1.-xi(3))/8.;
       N(4) = (1.-xi(1))*(1.+xi(2))*(1.-xi(3))/8.;
       N(5) = (1.-xi(1))*(1.-xi(2))*(1.+xi(3))/8.;
       N(6) = (1.+xi(1))*(1.-xi(2))*(1.+xi(3))/8.;
       N(7) = (1.+xi(1))*(1.+xi(2))*(1.+xi(3))/8.;
       N(8) = (1.-xi(1))*(1.+xi(2))*(1.+xi(3))/8.;
     elseif (nelnodes == 20) 
       N(1) = (1.-xi(1))*(1.-xi(2))*(1.-xi(3))*(-xi(1)-xi(2)-xi(3)-2.)/8.;
       N(2) = (1.+xi(1))*(1.-xi(2))*(1.-xi(3))*(xi(1)-xi(2)-xi(3)-2.)/8.;
       N(3) = (1.+xi(1))*(1.+xi(2))*(1.-xi(3))*(xi(1)+xi(2)-xi(3)-2.)/8.;
       N(4) = (1.-xi(1))*(1.+xi(2))*(1.-xi(3))*(-xi(1)+xi(2)-xi(3)-2.)/8.;
       N(5) = (1.-xi(1))*(1.-xi(2))*(1.+xi(3))*(-xi(1)-xi(2)+xi(3)-2.)/8.;
       N(6) = (1.+xi(1))*(1.-xi(2))*(1.+xi(3))*(xi(1)-xi(2)+xi(3)-2.)/8.;
       N(7) = (1.+xi(1))*(1.+xi(2))*(1.+xi(3))*(xi(1)+xi(2)+xi(3)-2.)/8.;
       N(8) = (1.-xi(1))*(1.+xi(2))*(1.+xi(3))*(-xi(1)+xi(2)+xi(3)-2.)/8.;
       N(9)  = (1.-xi(1)^2)*(1.-xi(2))*(1.-xi(3))/4.;
       N(10) = (1.+xi(1))*(1.-xi(2)^2)*(1.-xi(3))/4.;
       N(11) = (1.-xi(1)^2)*(1.+xi(2))*(1.-xi(3))/4.;
       N(12) = (1.-xi(1))*(1.-xi(2)^2)*(1.-xi(3))/4.;
       N(13) = (1.-xi(1)^2)*(1.-xi(2))*(1.+xi(3))/4.;
       N(14) = (1.+xi(1))*(1.-xi(2)^2)*(1.+xi(3))/4.;
       N(15) = (1.-xi(1)^2)*(1.+xi(2))*(1.+xi(3))/4.;
       N(16) = (1.-xi(1))*(1.-xi(2)^2)*(1.+xi(3))/4.;
       N(17) = (1.-xi(1))*(1.-xi(2))*(1.-xi(3)^2)/4.;
       N(18) = (1.+xi(1))*(1.-xi(2))*(1.-xi(3)^2)/4.;
       N(19) = (1.+xi(1))*(1.+xi(2))*(1.-xi(3)^2)/4.;
       N(20) = (1.-xi(1))*(1.+xi(2))*(1.-xi(3)^2)/4.;
     end
   end

end

%
%================= SHAPE FUNCTION DERIVATIVES ======================
%
function dNdxi = shapefunctionderivs(nelnodes,ncoord,elident,xi)

  dNdxi = zeros(nelnodes,ncoord);
%
% 1D elements
%
  if (ncoord == 1) 
    if (nelnodes==2) 
      dNdxi(1,1) = 0.5;
      dNdxi(2,1) = -0.5;
    elseif (nelnodes == 3) 
      dNdxi(1,1) = -0.5+xi(1);
      dNdxi(2,1) =  0.5+xi(1);
      dNdxi(3,1) = -2.*xi(1);
    end
%
%  2D elements
%
   elseif (ncoord == 2) 
%
%    Triangular element
%
     if ( nelnodes == 3 ) 
       dNdxi(1,1) = 1.;
       dNdxi(2,2) = 1.;
       dNdxi(3,1) = -1.;
       dNdxi(3,2) = -1.;               
     elseif ( nelnodes == 6 ) 
       xi3 = 1.-xi(1)-xi(2);
       dNdxi(1,1) = 4.*xi(1)-1.;
       dNdxi(2,2) = 4.*xi(2)-1.;
       dNdxi(3,1) = -(4.*xi3-1.);
       dNdxi(3,2) = -(4.*xi3-1.);
       dNdxi(4,1) = 4.*xi(2);
       dNdxi(4,2) = 4.*xi(1);
       dNdxi(5,1) = -4.*xi(2);
       dNdxi(5,2) = -4.*xi(1);
       dNdxi(6,1) = 4.*xi3 - 4.*xi(1);
       dNdxi(6,2) = 4.*xi3 - 4.*xi(2);
%
%    Rectangular element
%                  
     elseif ( nelnodes == 4 ) 
       dNdxi(1,1) = -0.25*(1.-xi(2));
       dNdxi(1,2) = -0.25*(1.-xi(1));
       dNdxi(2,1) = 0.25*(1.-xi(2));
       dNdxi(2,2) = -0.25*(1.+xi(1));
       dNdxi(3,1) = 0.25*(1.+xi(2));
       dNdxi(3,2) = 0.25*(1.+xi(1));
       dNdxi(4,1) = -0.25*(1.+xi(2));
       dNdxi(4,2) = 0.25*(1.-xi(1));
     elseif (nelnodes == 8) 
       dNdxi(1,1) = 0.25*(1.-xi(2))*(2.*xi(1)+xi(2));
       dNdxi(1,2) = 0.25*(1.-xi(1))*(xi(1)+2.*xi(2));
       dNdxi(2,1) = 0.25*(1.-xi(2))*(2.*xi(1)-xi(2));
       dNdxi(2,2) = 0.25*(1.+xi(1))*(2.*xi(2)-xi(1));
       dNdxi(3,1) = 0.25*(1.+xi(2))*(2.*xi(1)+xi(2));
       dNdxi(3,2) = 0.25*(1.+xi(1))*(2.*xi(2)+xi(1));
       dNdxi(4,1) = 0.25*(1.+xi(2))*(2.*xi(1)-xi(2));
       dNdxi(4,2) = 0.25*(1.-xi(1))*(2.*xi(2)-xi(1));
       dNdxi(5,1) = -xi(1)*(1.-xi(2));
       dNdxi(5,2) = -0.5*(1.-xi(1)*xi(1));
       dNdxi(6,1) = 0.5*(1.-xi(2)*xi(2));
       dNdxi(6,2) = -(1.+xi(1))*xi(2);
       dNdxi(7,1) = -xi(1)*(1.+xi(2));
       dNdxi(7,2) = 0.5*(1.-xi(1)*xi(1));
       dNdxi(8,1) = -0.5*(1.-xi(2)*xi(2));
       dNdxi(8,2) = -(1.-xi(1))*xi(2);
      end
%
%    3D elements
%
   elseif (ncoord==3) 

     if (nelnodes == 4) 
       dNdxi(1,1) = 1.;
       dNdxi(2,2) = 1.;
       dNdxi(3,3) = 1.;
       dNdxi(4,1) = -1.;
       dNdxi(4,2) = -1.;
       dNdxi(4,3) = -1.;
     elseif (nelnodes == 10) 
       xi4 = 1.-xi(1)-xi(2)-xi(3);
       dNdxi(1,1) = (4.*xi(1)-1.);
       dNdxi(2,2) = (4.*xi(2)-1.);
       dNdxi(3,3) = (4.*xi(3)-1.);
       dNdxi(4,1) = -(4.*xi4-1.);
       dNdxi(4,2) = -(4.*xi4-1.);
       dNdxi(4,3) = -(4.*xi4-1.);
       dNdxi(5,1) = 4.*xi(2);
       dNdxi(5,2) = 4.*xi(1);
       dNdxi(6,2) = 4.*xi(3);
       dNdxi(6,3) = 4.*xi(2);
       dNdxi(7,1) = 4.*xi(3);
       dNdxi(7,3) = 4.*xi(1); 
       dNdxi(8,1) = 4.*(xi4-xi(1));
       dNdxi(8,2) = -4.*xi(1);
       dNdxi(8,3) = -4.*xi(1);
       dNdxi(9,1) = -4.*xi(2);
       dNdxi(9,2) = 4.*(xi4-xi(2));
       dNdxi(9,3) = -4.*xi(2);
       dNdxi(10,1) = -4.*xi(3)*xi4;
       dNdxi(10,2) = -4.*xi(3);
       dNdxi(10,3) = 4.*(xi4-xi(3));
     elseif (nelnodes == 8) 
       dNdxi(1,1) = -(1.-xi(2))*(1.-xi(3))/8.;
       dNdxi(1,2) = -(1.-xi(1))*(1.-xi(3))/8.;
       dNdxi(1,3) = -(1.-xi(1))*(1.-xi(2))/8.;
       dNdxi(2,1) = (1.-xi(2))*(1.-xi(3))/8.;
       dNdxi(2,2) = -(1.+xi(1))*(1.-xi(3))/8.;
       dNdxi(2,3) = -(1.+xi(1))*(1.-xi(2))/8.;
       dNdxi(3,1) = (1.+xi(2))*(1.-xi(3))/8.;
       dNdxi(3,2) = (1.+xi(1))*(1.-xi(3))/8.;
       dNdxi(3,3) = -(1.+xi(1))*(1.+xi(2))/8.;
       dNdxi(4,1) = -(1.+xi(2))*(1.-xi(3))/8.;
       dNdxi(4,2) = (1.-xi(1))*(1.-xi(3))/8.;
       dNdxi(4,3) = -(1.-xi(1))*(1.+xi(2))/8.;
       dNdxi(5,1) = -(1.-xi(2))*(1.+xi(3))/8.;
       dNdxi(5,2) = -(1.-xi(1))*(1.+xi(3))/8.;
       dNdxi(5,3) = (1.-xi(1))*(1.-xi(2))/8.;
       dNdxi(6,1) = (1.-xi(2))*(1.+xi(3))/8.;
       dNdxi(6,2) = -(1.+xi(1))*(1.+xi(3))/8.;
       dNdxi(6,3) = (1.+xi(1))*(1.-xi(2))/8.;
       dNdxi(7,1) = (1.+xi(2))*(1.+xi(3))/8.;
       dNdxi(7,2) = (1.+xi(1))*(1.+xi(3))/8.;
       dNdxi(7,3) = (1.+xi(1))*(1.+xi(2))/8.;
       dNdxi(8,1) = -(1.+xi(2))*(1.+xi(3))/8.;
       dNdxi(8,2) = (1.-xi(1))*(1.+xi(3))/8.;
       dNdxi(8,3) = (1.-xi(1))*(1.+xi(2))/8.;
     elseif (nelnodes == 20) 
       dNdxi(1,1) = (-(1.-xi(2))*(1.-xi(3))*(-xi(1)-xi(2)-xi(3)-2.)-(1.-xi(1))*(1.-xi(2))*(1.-xi(3)))/8.;
       dNdxi(1,2) = (-(1.-xi(1))*(1.-xi(3))*(-xi(1)-xi(2)-xi(3)-2.)-(1.-xi(1))*(1.-xi(2))*(1.-xi(3)))/8.;
       dNdxi(1,3) = (-(1.-xi(1))*(1.-xi(2))*(-xi(1)-xi(2)-xi(3)-2.)-(1.-xi(1))*(1.-xi(2))*(1.-xi(3)))/8.;

       dNdxi(2,1) = ((1.-xi(2))*(1.-xi(3))*(xi(1)-xi(2)-xi(3)-2.)+(1.+xi(1))*(1.-xi(2))*(1.-xi(3)))/8.;
       dNdxi(2,2) = (-(1.+xi(1))*(1.-xi(3))*(xi(1)-xi(2)-xi(3)-2.)-(1.+xi(1))*(1.-xi(2))*(1.-xi(3)))/8.;
       dNdxi(2,3) = (-(1.+xi(1))*(1.-xi(2))*(xi(1)-xi(2)-xi(3)-2.)-(1.+xi(1))*(1.-xi(2))*(1.-xi(3)))/8.;

       dNdxi(3,1) = ((1.+xi(2))*(1.-xi(3))*(xi(1)+xi(2)-xi(3)-2.)+(1.+xi(1))*(1.+xi(2))*(1.-xi(3)))/8.;
       dNdxi(3,2) = ((1.+xi(1))*(1.-xi(3))*(xi(1)+xi(2)-xi(3)-2.)+(1.+xi(1))*(1.+xi(2))*(1.-xi(3)))/8.;
       dNdxi(3,3) = (-(1.+xi(1))*(1.+xi(2))*(xi(1)+xi(2)-xi(3)-2.)-(1.+xi(1))*(1.+xi(2))*(1.-xi(3)))/8.;

       dNdxi(4,1) = (-(1.+xi(2))*(1.-xi(3))*(-xi(1)+xi(2)-xi(3)-2.)-(1.-xi(1))*(1.+xi(2))*(1.-xi(3)))/8.;
       dNdxi(4,2) = ((1.-xi(1))*(1.-xi(3))*(-xi(1)+xi(2)-xi(3)-2.)+(1.-xi(1))*(1.+xi(2))*(1.-xi(3)))/8.;
       dNdxi(4,3) = (-(1.-xi(1))*(1.+xi(2))*(-xi(1)+xi(2)-xi(3)-2.)-(1.-xi(1))*(1.+xi(2))*(1.-xi(3)))/8.;
       dNdxi(5,1) = (-(1.-xi(2))*(1.+xi(3))*(-xi(1)-xi(2)+xi(3)-2.)-(1.-xi(1))*(1.-xi(2))*(1.+xi(3)))/8.;
       dNdxi(5,2) = (-(1.-xi(1))*(1.+xi(3))*(-xi(1)-xi(2)+xi(3)-2.)-(1.-xi(1))*(1.-xi(2))*(1.+xi(3)))/8.;
       dNdxi(5,3) = ((1.-xi(1))*(1.-xi(2))*(-xi(1)-xi(2)+xi(3)-2.)+(1.-xi(1))*(1.-xi(2))*(1.+xi(3)))/8.;
       dNdxi(6,1) = ((1.-xi(2))*(1.+xi(3))*(xi(1)-xi(2)+xi(3)-2.)+(1.+xi(1))*(1.-xi(2))*(1.+xi(3)))/8.;
       dNdxi(6,2) = (-(1.+xi(1))*(1.+xi(3))*(xi(1)-xi(2)+xi(3)-2.)-(1.+xi(1))*(1.-xi(2))*(1.+xi(3)))/8.;
       dNdxi(6,3) = ((1.+xi(1))*(1.-xi(2))*(xi(1)-xi(2)+xi(3)-2.)+(1.+xi(1))*(1.-xi(2))*(1.+xi(3)))/8.;
       dNdxi(7,1) = ((1.+xi(2))*(1.+xi(3))*(xi(1)+xi(2)+xi(3)-2.)+(1.+xi(1))*(1.+xi(2))*(1.+xi(3)))/8.;
       dNdxi(7,2) = ((1.+xi(1))*(1.+xi(3))*(xi(1)+xi(2)+xi(3)-2.)+(1.+xi(1))*(1.+xi(2))*(1.+xi(3)))/8.;
       dNdxi(7,3) = ((1.+xi(1))*(1.+xi(2))*(xi(1)+xi(2)+xi(3)-2.)+(1.+xi(1))*(1.+xi(2))*(1.+xi(3)))/8.;
       dNdxi(8,1) = (-(1.+xi(2))*(1.+xi(3))*(-xi(1)+xi(2)+xi(3)-2.)-(1.-xi(1))*(1.+xi(2))*(1.+xi(3)))/8.;
       dNdxi(8,2) = ((1.-xi(1))*(1.+xi(3))*(-xi(1)+xi(2)+xi(3)-2.)+(1.-xi(1))*(1.+xi(2))*(1.+xi(3)))/8.;
       dNdxi(8,3) = ((1.-xi(1))*(1.+xi(2))*(-xi(1)+xi(2)+xi(3)-2.)+(1.-xi(1))*(1.+xi(2))*(1.+xi(3)))/8.;
       dNdxi(9,1)  = -2.*xi(1)*(1.-xi(2))*(1.-xi(3))/4.;
       dNdxi(9,2)  = -(1.-xi(1)^2)*(1.-xi(3))/4.;
       dNdxi(9,3)  = -(1.-xi(1)^2)*(1.-xi(2))/4.;
       dNdxi(10,1)  = (1.-xi(2)^2)*(1.-xi(3))/4.;
       dNdxi(10,2)  = -2.*xi(2)*(1.+xi(1))*(1.-xi(3))/4.;
       dNdxi(10,3)  = -(1.-xi(2)^2)*(1.+xi(1))/4.;
       dNdxi(11,1)  = -2.*xi(1)*(1.+xi(2))*(1.-xi(3))/4.;
       dNdxi(11,2)  = (1.-xi(1)^2)*(1.-xi(3))/4.;
       dNdxi(11,3)  = -(1.-xi(1)^2)*(1.+xi(2))/4.;
       dNdxi(12,1)  = -(1.-xi(2)^2)*(1.-xi(3))/4.;
       dNdxi(12,2)  = -2.*xi(2)*(1.-xi(1))*(1.-xi(3))/4.;
       dNdxi(12,3)  = -(1.-xi(2)^2)*(1.-xi(1))/4.;
       dNdxi(13,1)  = -2.*xi(1)*(1.-xi(2))*(1.+xi(3))/4.;
       dNdxi(13,2)  = -(1.-xi(1)^2)*(1.+xi(3))/4.;
       dNdxi(13,3)  = (1.-xi(1)^2)*(1.-xi(2))/4.;
       dNdxi(14,1)  = (1.-xi(2)^2)*(1.+xi(3))/4.;
       dNdxi(14,2)  = -2.*xi(2)*(1.+xi(1))*(1.+xi(3))/4.;
       dNdxi(14,3)  = (1.-xi(2)^2)*(1.+xi(1))/4.;
       dNdxi(15,1)  = -2.*xi(1)*(1.+xi(2))*(1.+xi(3))/4.;
       dNdxi(15,2)  =  (1.-xi(1)^2)*(1.+xi(3))/4.;
       dNdxi(15,3)  = (1.-xi(1)^2)*(1.+xi(2))/4.;
       dNdxi(16,1)  = -(1.-xi(2)^2)*(1.+xi(3))/4.;
       dNdxi(16,2)  = -2.*xi(2)*(1.-xi(1))*(1.+xi(3))/4.;
       dNdxi(16,3)  = (1.-xi(2)^2)*(1.-xi(1))/4.;
       dNdxi(17,1) = -(1.-xi(2))*(1.-xi(3)^2)/4.;
       dNdxi(17,2) = -(1.-xi(1))*(1.-xi(3)^2)/4.;
       dNdxi(17,3) = -xi(3)*(1.-xi(1))*(1.-xi(2))/2.;
       dNdxi(18,1) = (1.-xi(2))*(1.-xi(3)^2)/4.;
       dNdxi(18,2) = -(1.+xi(1))*(1.-xi(3)^2)/4.;
       dNdxi(18,3) = -xi(3)*(1.+xi(1))*(1.-xi(2))/2.;
       dNdxi(19,1) = (1.+xi(2))*(1.-xi(3)^2)/4.;
       dNdxi(19,2) = (1.+xi(1))*(1.-xi(3)^2)/4.;
       dNdxi(19,3) = -xi(3)*(1.+xi(1))*(1.+xi(2))/2.;
       dNdxi(20,1) = -(1.+xi(2))*(1.-xi(3)^2)/4.;
       dNdxi(20,2) = (1.-xi(1))*(1.-xi(3)^2)/4.;
       dNdxi(20,3) = -xi(3)*(1.-xi(1))*(1.+xi(2))/2.;
     end
   end

end
%
%================= ELEMENT RESIDUAL VECTOR ================================
%
function rel = elresid(dt,ncoord,ndof,nelnodes,elident,coord,materialprops,stress0,eplas0,displacement)
%
%  Assemble the element residual force
%
%    Arguments:
%
%      ncoord             No. coordinates (2 or 3 for 2D or 3D problem)
%      ndof               No. degrees of freedom per node (often ndof = ncoord)
%      nelnodes           No. nodes on the element
%      elident            Element identifier (not used here - for future enhancements!)
%      coords[i,a]        ith coord of ath node
%      materialprops      Material properties passed on to constitutive procedures
%      displacement[i,a]  ith displacement component at ath node
%
%   Local variables
%      npoints            No. integration points
%      xi[i,inpt]         ith local coord of integration point no. intpt
%      w[intpt]           weight for integration point no. intpt
%      N[a]               Shape function associated with ath node on element
%      dNdxi[a,i]         Derivative of ath shape function wrt ith local coord
%      dNdx[a,i]          Derivative of ath shape function wrt ith global coord
%      dxdxi[i,j]         Derivative of ith global coord wrt jth local coord
%      dxidx[i,j]         Derivative of ith local coord wrt jth global coord
%      det                Determinant of jacobian
%      strain[i,j]        strain_ij components
%      stress[i,j]        stress_ij components
%      r[row]             Residual vector
%
%
   npoints = numberofintegrationpoints(ncoord,nelnodes,elident);
   strain = zeros(3,3);
   dxdxi = zeros(ncoord,ncoord);
   dxidx = zeros(ncoord,ncoord);
   rel = zeros(ndof*nelnodes,1);
   stressi0 = zeros(3,3);
%
%  Set up integration points and weights    
%
   xilist = integrationpoints(ncoord,nelnodes,npoints,elident);
   w = integrationweights(ncoord,nelnodes,npoints,elident);
   %
%  Loop over the integration points
%
   for intpt = 1 : npoints

%     Compute shape functions and derivatives wrt local coords
%
      for i = 1 : ncoord
        xi(i) = xilist(i,intpt);
      end      
      N = shapefunctions(nelnodes,ncoord,elident,xi);
      dNdxi = shapefunctionderivs(nelnodes,ncoord,elident,xi);
%
%     Compute the jacobian matrix and its determinant
%
      for i = 1 : ncoord
        for j = 1 : ncoord
          dxdxi(i,j) = 0.;
          for a = 1 : nelnodes
            dxdxi(i,j) = dxdxi(i,j) + coord(i,a)*dNdxi(a,j);
          end
        end
      end

      dxidx = inv(dxdxi);
      dtm = det(dxdxi);
%
%     Convert shape function derivatives to derivatives wrt global coords
%
      for a = 1 : nelnodes
        for i = 1 : ncoord
          dNdx(a,i) = 0.;
          for j = 1 : ncoord
            dNdx(a,i) = dNdx(a,i) + dNdxi(a,j)*dxidx(j,i);
          end
        end
      end
%
%     Compute the (infinitesimal) strain by differentiating displacements
%
      for i = 1 : ncoord
         for j = 1 : ncoord
            strain(i,j) = 0.;
            for a = 1 : nelnodes
              strain(i,j) = strain(i,j) ... 
                         + 0.5*(displacement(i,a)*dNdx(a,j)+displacement(j,a)*dNdx(a,i));
            end
         end
      end
%
%     Compute the stress
%
      for i = 1 : 3
        for j = 1 : 3
          stressi0(i,j) = stress0(i,j,intpt);
        end
      end
%     Compute the material tangent stiffness (d stress/d strain)
%
      ep0 = eplas0(intpt);
      dep = deplas(dt,stressi0,ep0,strain,materialprops);
      stress = materialstress(dt,ndof,ncoord,stressi0,ep0,dep,strain,materialprops);
%
%     Compute the element residual
%             
      for a = 1 : nelnodes
        for i = 1 : ndof
          row = ndof*(a-1)+i;
          for j = 1 : ncoord 
            rel(row) = rel(row) + stress(i,j)*dNdx(a,j)*w(intpt)*dtm;
          end
        end
      end
   end

end
%
%================= ELEMENT STIFFNESS MATRIX ================================
%
function kel = elstif(dt,ncoord,ndof,nelnodes,elident,coord,materialprops,stress0,eplas0,displacement)
%
%  Assemble the element stiffness
%
%    Arguments;
%
%      ncoord             No. coordinates (2 or 3 for 2D or 3D problem)
%      ndof               No. degrees of freedom per node (often ndof = ncoord)
%      nelnodes           No. nodes on the element
%      elident            Element identifier (not used here - for future enhancements!)
%      coords(i,a)        ith coord of ath node
%      materialprops      Material properties passed on:constitutive procedures
%      displacement(i,a)  ith displacement component at ath node
%
%   Local variables
%      npoints            No. integration points
%      xi(i,inpt)         ith local coord of integration point no. intpt
%      w(intpt)           weight for integration point no. intpt
%      N(a)               Shape function associated with ath node on element
%      dNdxi(a,i)         Derivative of ath shape function wrt ith local coord
%      dNdx(a,i)          Derivative of ath shape function wrt ith global coord
%      dxdxi(i,j)         Derivative of ith global coord wrt jth local coord
%      dxidx(i,j)         Derivative of ith local coord wrt jth global coord
%      det                Determinant of jacobian
%      strain(i,j)        strain_ij components
%      dsde(i,j,k,l)      Derivative of stress_ij with respect:strain_kl
%      kel(row,col)       Rows && cols of element stiffness
%
%
   npoints = numberofintegrationpoints(ncoord,nelnodes,elident);
   dNdx = zeros(nelnodes,ncoord);
   dxdxi = zeros(ncoord,ncoord);
   strain = zeros(3,3);
   kel = zeros(ndof*nelnodes,ndof*nelnodes);
%
%  Set up integration points && weights    
%
   xilist = integrationpoints(ncoord,nelnodes,npoints,elident);
   w = integrationweights(ncoord,nelnodes,npoints,elident);
   stressi0 = zeros(3,3);
%
%  Loop over the integration points
%
   for intpt = 1:npoints

%     Compute shape functions && derivatives wrt local coords
%
      for i = 1:ncoord
        xi(i) = xilist(i,intpt);
      end      
      N = shapefunctions(nelnodes,ncoord,elident,xi);
      dNdxi = shapefunctionderivs(nelnodes,ncoord,elident,xi);

      
%
%     Compute the jacobian matrix && its determinant
%
      for i = 1:ncoord
        for j = 1:ncoord
          dxdxi(i,j) = 0.;
          for a = 1:nelnodes
            dxdxi(i,j) = dxdxi(i,j) + coord(i,a)*dNdxi(a,j);
          end
        end
      end
      
      dxidx = inv(dxdxi);
      dtm = det(dxdxi);
%
%     Convert shape function derivatives:derivatives wrt global coords
%
      for a = 1:nelnodes
        for i = 1:ncoord
          dNdx(a,i) = 0.;
          for j = 1:ncoord
            dNdx(a,i) = dNdx(a,i) + dNdxi(a,j)*dxidx(j,i);
          end
        end
      end
%
%     Compute the (infinitesimal) strain by differentiating displacements
%     This step is not really necessary for linear elasticity calculations
%     where stiffness is independent of strain.  It is included:allow
%     extension:nonlinear materials later.
%
      for i = 1:ncoord
         for j = 1:ncoord
            strain(i,j) = 0.;
            for a = 1:nelnodes
              strain(i,j) = strain(i,j) + 0.5*(displacement(i,a)*dNdx(a,j)+displacement(j,a)*dNdx(a,i));
            end
         end
      end
%
      for i = 1 : 3
        for j = 1 : 3
          stressi0(i,j) = stress0(i,j,intpt);
        end
      end
%     Compute the material tangent stiffness (d stress/d strain)
%
      ep0 = eplas0(intpt);
      dep = deplas(dt,stressi0,ep0,strain,materialprops);
      stress = materialstress(dt,ndof,ncoord,stressi0,ep0,dep,strain,materialprops);
      dsde = materialstiffness(ndof,ncoord,stress,ep0,dep,strain,materialprops);
%
%     Compute the element stiffness
%             
      for a = 1:nelnodes
        for i = 1:ndof
          for b = 1:nelnodes
            for k = 1:ndof
              row = ndof*(a-1)+i;
              col = ndof*(b-1)+k;
              for j = 1:ncoord
                for l = 1:ncoord
                  kel(col,row) = kel(col,row) + dsde(i,j,k,l)*dNdx(b,l)*dNdx(a,j)*w(intpt)*dtm;
                end
              end
            end
          end
        end
      end
   end

end


 %====================== No. nodes on element faces ================
%
%   This procedure returns the number of nodes on each element face
%   for various element types.  This info is needed for computing
%   the surface integrals associated with the element traction vector
%
function n = nfacenodes(ncoord,nelnodes,elident,face)
   if (ncoord == 2) 
     if (nelnodes == 3 || nelnodes == 4)
         n = 2;
     elseif (nelnodes == 6 || nelnodes == 8)
         n=3;
     end
   elseif (ncoord == 3) 
     if (nelnodes == 4)
         n = 3;
     elseif (nelnodes == 10)
         n = 6;
     elseif (nelnodes == 8)
         n = 4;
     elseif (nelnodes == 20)
         n = 8;
     end
   end
end
%======================= Lists of nodes on element faces =============
%
%    This procedure returns the list of nodes on an element face
%    The nodes are ordered so that the element face forms either
%    a 1D line element or a 2D surface element for 2D or 3D problems
%
function list = facenodes(ncoord,nelnodes,elident,face)

   i3 = [2,3,1];
   i4 = [2,3,4,1]; 

   list = zeros(nfacenodes(ncoord,nelnodes,face),1);

   if (ncoord == 2) 
     if (nelnodes == 3) 
       list(1) = face;
       list(2) = i3(face);
     elseif (nelnodes == 6) 
       list(1) = face;
       list(2) = i3(face);
       list(3) = face+3;
     elseif (nelnodes==4) 
       list(1) = face;
       list(2) = i4(face);
     elseif (nelnodes==8) 
       list(1) = face;
       list(2) = i4(face);
       list(3) = face+4;
     end
   elseif (ncoord == 3) 
     if (nelnodes==4) 
       if   (face == 1)
           list = [1,2,3];
       elseif (face == 2)
           list = [1,4,2];
       elseif (face == 3)
           list = [2,4,3];
       elseif (face == 4)
           list = [3,4,1];
       end
     elseif (nelnodes == 10) 
       if   (face == 1)
           list = [1,2,3,5,6,7];
       elseif (face == 2)
           list = [1,4,2,8,9,5];
       elseif (face == 3)
           list = [2,4,3,9,10,6];
       elseif (face == 4)
           list = [3,4,1,10,8,7];
       end
     elseif (nelnodes == 8) 
       if   (face == 1)
           list = [1,2,3,4];
       elseif (face == 2)
           list = [5,8,7,6];
       elseif (face == 3)
           list = [1,5,6,2];
       elseif (face == 4)
           list = [2,3,7,6];
       elseif (face == 5)
           list = [3,7,8,4];
       elseif (face == 6)
           list = [4,8,5,1];
       end
     elseif (nelnodes == 20)  
       if   (face == 1)
           list = [1,2,3,4,9,10,11,12];
       elseif (face == 2)
           list = [5,8,7,6,16,15,14,13];
       elseif (face == 3)
           list = [1,5,6,2,17,13,18,9];
       elseif (face == 4)
           list = [2,6,7,3,18,14,19,10];
       elseif (face == 5)
           list = [3,7,8,4,19,15,20,11];
       elseif (face == 6)
           list = [4,8,5,1,20,16,17,12];
       end
     end
   end
end
%
%====================== ELEMENT DISTRIBUTED LOAD VECTOR ==============
%
function r = eldload(ncoord,ndof,nfacenodes,elident,coords,traction)

  npoints = numberofintegrationpoints(ncoord-1,nfacenodes);
  xi = zeros(ncoord-1,1);
  dxdxi = zeros(ncoord,ncoord-1);
  r = zeros(ndof*nfacenodes,1);
   
  xilist = integrationpoints(ncoord-1,nfacenodes,npoints);
  w = integrationweights(ncoord-1,nfacenodes,npoints);

  for intpt = 1:npoints

    for i = 1:ncoord-1
      xi(i) = xilist(i,intpt);
    end

    N = shapefunctions(nfacenodes,ncoord-1,elident,xi);
    dNdxi = shapefunctionderivs(nfacenodes,ncoord-1,elident,xi);
%
%     Compute the jacobian matrix && its determinant
%
    for i = 1:ncoord
      for j = 1:ncoord-1
        dxdxi(i,j) = 0.;
        for a = 1:nfacenodes
          dxdxi(i,j) = dxdxi(i,j) + coords(i,a)*dNdxi(a,j);
        end
      end
    end
    if (ncoord == 2) 
      dt = sqrt(dxdxi(1,1)^2+dxdxi(2,1)^2);
    elseif (ncoord == 3) 
      dt = sqrt( ((dxdxi(2,1)*dxdxi(3,2))-(dxdxi(2,2)*dxdxi(3,1)))^2 ...
          + ((dxdxi(1,1)*dxdxi(3,2))-(dxdxi(1,2)*dxdxi(3,1)))^2 ...
          + ((dxdxi(1,1)*dxdxi(2,2))-(dxdxi(1,2)*dxdxi(2,1)))^2 );
    end
   for a = 1:nfacenodes
      for i = 1:ndof
        row = ndof*(a-1)+i;
        r(row) = r(row) + N(a)*traction(i)*w(intpt)*dt;
      end
    end
  end
end
%
%================= UPDATE_EL_STATE ================================
%
function [stress1,ep1] = update_el_state(dt,ncoord,ndof,nelnodes,elident,coord,materialprops,stress0,eplas0,displacement)
%
%  Compute the stress and accumulated plastic strain at the end of a load
%  increment at all integration points in an element
%
%    Arguments:
%
%      ncoord             No. coordinates (2 or 3 for 2D or 3D problem)
%      ndof               No. degrees of freedom per node (often ndof = ncoord)
%      nelnodes           No. nodes on the element
%      elident            Element identifier (not used here - for future enhancements!)
%      coords[i,a]        ith coord of ath node
%      materialprops      Material properties passed on to constitutive procedures
%      displacement[i,a]  ith displacement component at ath node
%
%   Local variables
%      npoints            No. integration points
%      xi[i,inpt]         ith local coord of integration point no. intpt
%      w[intpt]           weight for integration point no. intpt
%      N[a]               Shape function associated with ath node on element
%      dNdxi[a,i]         Derivative of ath shape function wrt ith local coord
%      dNdx[a,i]          Derivative of ath shape function wrt ith global coord
%      dxdxi[i,j]         Derivative of ith global coord wrt jth local coord
%      dxidx[i,j]         Derivative of ith local coord wrt jth global coord
%      det                Determinant of jacobian
%      strain[i,j]        strain_ij components
%      stress[i,j]        stress_ij components
%      r[row]             Residual vector
%
%
   npoints = numberofintegrationpoints(ncoord,nelnodes,elident);
   dxdxi = zeros(ncoord,ncoord);
   dxidx = zeros(ncoord,ncoord);
   strain = zeros(3,3);
   stressi0 = zeros(3,3);
%
%  Set up integration points and weights    
%
   xilist = integrationpoints(ncoord,nelnodes,npoints,elident);
   w = integrationweights(ncoord,nelnodes,npoints,elident);
   %
%  Loop over the integration points
%
   for intpt = 1 : npoints

%     Compute shape functions and derivatives wrt local coords
%
      for i = 1 : ncoord
        xi(i) = xilist(i,intpt);
      end      
      N = shapefunctions(nelnodes,ncoord,elident,xi);
      dNdxi = shapefunctionderivs(nelnodes,ncoord,elident,xi);
%
%     Compute the jacobian matrix and its determinant
%
      for i = 1 : ncoord
        for j = 1 : ncoord
          dxdxi(i,j) = 0.;
          for a = 1 : nelnodes
            dxdxi(i,j) = dxdxi(i,j) + coord(i,a)*dNdxi(a,j);
          end
        end
      end

      dxidx = inv(dxdxi);
      dtm = det(dxdxi);
%
%     Convert shape function derivatives to derivatives wrt global coords
%
      for a = 1 : nelnodes
        for i = 1 : ncoord
          dNdx(a,i) = 0.;
          for j = 1 : ncoord
            dNdx(a,i) = dNdx(a,i) + dNdxi(a,j)*dxidx(j,i);
          end
        end
      end
%
%     Compute the (infinitesimal) strain by differentiating displacements
%
      for i = 1 : ncoord
         for j = 1 : ncoord
            strain(i,j) = 0.;
            for a = 1 : nelnodes
              strain(i,j) = strain(i,j) ... 
                         + 0.5*(displacement(i,a)*dNdx(a,j)+displacement(j,a)*dNdx(a,i));
            end
         end
      end
%
%     Compute the stress
%
      for i = 1 : 3
        for j = 1 : 3
          stressi0(i,j) = stress0(i,j,intpt);
        end
      end
%     Compute the material tangent stiffness (d stress/d strain)
%
      ep0 = eplas0(intpt);
      dep = deplas(dt,stressi0,ep0,strain,materialprops);
      stress = materialstress(dt,ndof,ncoord,stressi0,ep0,dep,strain,materialprops);
      
      for i = 1 : 3
        for j = 1 : 3
          stress1(i,j,intpt) = stress(i,j);
        end
      end
      ep1(intpt) = ep0 + dep;
      
      
   end

end


%
%====================== Assemble the global residual vector =================
%
function resid = globalresidual(dt,ncoord,ndof,nnode,coords,nelem,maxnodes,elident,nelnodes,connect,materialprops,stress,eplas,dofs)
%
%   Assemble the global stiffness matrix
%

   resid = zeros(ndof*nnode,1);
   lmncoord = zeros(ncoord,maxnodes);
   lmndof = zeros(ndof,maxnodes);
   rel = zeros(ndof*maxnodes,ndof*maxnodes);
%
%   Loop over all the elements
%
   for lmn = 1 : nelem
%
%   Extract coords of nodes, DOF for the current element
%
      for a = 1 : nelnodes(lmn)
        for i = 1 : ncoord
          lmncoord(i,a) = coords(i,connect(a,lmn));
        end
        for i = 1 : ndof
          lmndof(i,a) = dofs(ndof*(connect(a,lmn)-1)+i);
        end
      end
      n = nelnodes(lmn);
      ident = elident(lmn);
      nintp = numberofintegrationpoints(ncoord,n,ident);
      lmnstress = zeros(3,3,nintp);
      lmneplas = zeros(nintp,1);
      for a = 1 : nintp
        for i = 1 : 3
          for j = 1 : 3
            lmnstress(i,j,a) = stress(i,j,a,lmn);
          end
        end
        lmneplas(a) = eplas(a,lmn);
      end    
      
      rel = elresid(dt,ncoord,ndof,n,ident,lmncoord,materialprops,lmnstress,lmneplas,lmndof);

%
%   Add the current element residual to the global residual
%
      for a = 1 : nelnodes(lmn)
        for i = 1 : ndof
          rw = ndof*(connect(a,lmn)-1)+i;
          resid(rw) = resid(rw) + rel(ndof*(a-1)+i);
        end
     end
  end
 
end
%
%====================== Assemble the global stiffness matrix =================
%
function Stif = globalstiffness(dt,ncoord,ndof,nnode,coords,nelem,maxnodes,elident,nelnodes,connect,materialprops,stress,eplas,dofs)
%
%   Assemble the global stiffness matrix
%
 
   Stif = zeros(ndof*nnode,ndof*nnode);
   lmncoord = zeros(ncoord,maxnodes);
   lmndof = zeros(ndof,maxnodes);
%
%   Loop over all the elements
%
   for lmn = 1:nelem
%
%   Extract coords of nodes, DOF for the current element
%
      for a = 1:nelnodes(lmn)
        for i = 1:ncoord
          lmncoord(i,a) = coords(i,connect(a,lmn));
        end
        for i = 1:ndof
          lmndof(i,a) = dofs(ndof*(connect(a,lmn)-1)+i);
        end
      end
      n = nelnodes(lmn);
      ident = elident(lmn);
      nintp = numberofintegrationpoints(ncoord,n,ident);
      lmnstress = zeros(3,3,nintp);
      lmneplas = zeros(nintp,1);
      for a = 1 : nintp
        for i = 1 : 3
          for j = 1 : 3
            lmnstress(i,j,a) = stress(i,j,a,lmn);
          end
        end
        lmneplas(a) = eplas(a,lmn);
      end  
      kel = elstif(dt,ncoord,ndof,n,ident,lmncoord,materialprops,lmnstress,lmneplas,lmndof);
%
%   Add the current element stiffness:the global stiffness
%
      for a = 1:nelnodes(lmn)
        for i = 1:ndof
          for b = 1:nelnodes(lmn)
            for k = 1:ndof
              rw = ndof*(connect(a,lmn)-1)+i;
              cl = ndof*(connect(b,lmn)-1)+k;
              Stif(rw,cl) = Stif(rw,cl) + kel(ndof*(a-1)+i,ndof*(b-1)+k);
            end
          end
        end
      end
   end
end
%
%===================== Assemble the global traction vector =============
%
function r = globaltraction(ncoord,ndof,nnodes,ndload,coords,nelnodes,elident,connect,dloads,dofs)

   r = zeros(ndof*nnodes,1);
   traction = zeros(ndof,1);

   for load = 1:ndload
%
%     Extract the coords of the nodes on the appropriate element face
%
      lmn = dloads(1,load);
      face = dloads(2,load);
      n = nelnodes(lmn);
      ident = elident(lmn);
      nfnodes = nfacenodes(ncoord,n,ident,face); 
      nodelist = facenodes(ncoord,n,ident,face);     
      lmncoord = zeros(ncoord,nfnodes);
      for a = 1:nfnodes
        for i = 1:ncoord
          lmncoord(i,a) = coords(i,connect(nodelist(a),dloads(1,load)));
        end
        for i = 1:ndof
          lmndof(i,a) = dofs(ndof*(connect(nodelist(a),dloads(1,load))-1)+i);
        end
      end
%
%    Compute the element load vector
%
     for i = 1:ndof
       traction(i) = dloads(i+2,load);
     end

     rel = eldload(ncoord,ndof,nfnodes,ident,lmncoord,traction);
%
%    Assemble the element load vector into global vector
%
     for a = 1:nfnodes
       for i = 1:ndof
         rw = (connect(nodelist(a),dloads(1,load))-1)*ndof+i;
         r(rw) = r(rw) + rel((a-1)*ndof+i);
       end
     end

   end
end       
%
%====================== Update state variables =================
%
function [stress_new,eplas_new] = update_state(dt,ncoord,ndof,nnode,coords,nelem,maxnodes,elident,nelnodes,connect,materialprops,stress,eplas,dofs)
%
%   Assemble the global stiffness matrix
%

   lmncoord = zeros(ncoord,maxnodes);
   lmndof = zeros(ndof,maxnodes);
%
%   Loop over all the elements
%
   for lmn = 1 : nelem
%
%   Extract coords of nodes, DOF for the current element
%
      for a = 1 : nelnodes(lmn)
        for i = 1 : ncoord
          lmncoord(i,a) = coords(i,connect(a,lmn));
        end
        for i = 1 : ndof
          lmndof(i,a) = dofs(ndof*(connect(a,lmn)-1)+i);
        end
      end
      n = nelnodes(lmn);
      ident = elident(lmn);
      nintp = numberofintegrationpoints(ncoord,n,ident);
      lmnstress = zeros(3,3,nintp);
      lmneplas = zeros(nintp,1);
      for a = 1 : nintp
        for i = 1 : 3
          for j = 1 : 3
            lmnstress(i,j,a) = stress(i,j,a,lmn);
          end
        end
        lmneplas(a) = eplas(a,lmn);
      end    
      
      [lmnstress,lmneplas] = update_el_state(dt,ncoord,ndof,n,ident,lmncoord,materialprops,lmnstress,lmneplas,lmndof);
%
      for a = 1 : nintp
        for i = 1 : 3
          for j = 1 : 3
            stress_new(i,j,a,lmn) = lmnstress(i,j,a);
          end
        end
        eplas_new(a,lmn) = lmneplas(a);
      end    
      

  end
 
end
%================================== Print_results ======================
function print_results(outfile, ...
    nprops,materialprops,ncoord,ndof,nnode,coords, ...
    nelem,maxnodes,connect,nelnodes,elident,stress,eplas, ...
    nfix,fixnodes,ndload,dloads,dofs)
%      Print nodal displacements, element strains && stresses:a file

fprintf(outfile,'Nodal Displacements: \n');
   if (ndof == 2) 
     fprintf(outfile,' Node      Coords         u1       u2 \n');
     for i = 1:nnode
      fprintf(outfile,'%3d %8.4f %8.4f %8.4f %8.4f\n', ...
                               i,coords(1,i),coords(2,i),dofs(2*i-1),dofs(2*i));
     end
   elseif (ndof == 3) 
     fprintf(outfile,' Node            Coords            u1       u2       u3 \n');
     for i = 1:nnode
      fprintf(outfile,'%3d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f \n', ...
                    i,coords(1,i),coords(2,i),coords(3,i),dofs(3*i-2),dofs(3*i-1),dofs(3*i));
     end
   end

   fprintf(outfile,'\n\n Strains and Stresses \n');


   lmncoord = zeros(ncoord,maxnodes);
   displacements = zeros(ndof,maxnodes);

%
%   Loop over all the elements
%
   for lmn = 1:nelem

    fprintf(outfile,' \n Element; %d ',lmn);
    if (ncoord == 2)   
    fprintf(outfile,'  \n int pt    Coords      eplas         e_11      e_22     e_12      s_11       s_22      s_12 \n');

    elseif (ncoord == 3) 
    fprintf(outfile,'\n int pt         Coords     eplas         e_11      e_22     e_33      e_12       e_13      e_23      s_11      s_22      s_33      s_12      s_13      s_23 \n');
    end
%
%   Extract coords of nodes, DOF for the current element
%
      for a = 1:nelnodes(lmn)
        for i = 1:ncoord
          lmncoord(i,a) = coords(i,connect(a,lmn));
        end
        for i = 1:ndof
          displacements(i,a) = dofs(ndof*(connect(a,lmn)-1)+i);
        end
      end
      n = nelnodes(lmn);
      ident = elident(lmn);
 
      npoints = numberofintegrationpoints(ncoord,n);
      dNdx = zeros(n,ncoord);
      dxdxi = zeros(ncoord,ncoord);
      strain = zeros(ndof,ncoord);
      xi = zeros(ncoord,1);
      x = zeros(ncoord,1);
%
%  Set up integration points 
%
      xilist = integrationpoints(ncoord,n,npoints);
%
%  Loop over the integration points
%
     for intpt = 1:npoints

%     Compute shape functions && derivatives wrt local coords
%
       for i = 1:ncoord
         xi(i) = xilist(i,intpt);
       end
       N = shapefunctions(n,ncoord,ident,xi);      
       dNdxi = shapefunctionderivs(n,ncoord,ident,xi);
%
%     Compute the coords of the integration point
%
      for i = 1:ncoord
        x(i) = 0.;
        for a = 1:n
          x(i) = x(i) + lmncoord(i,a)*N(a);
        end
      end
%
%     Compute the jacobian matrix && its determinant
%
      for i = 1:ncoord
        for j = 1:ncoord
          dxdxi(i,j) = 0.;
          for a = 1:n
            dxdxi(i,j) = dxdxi(i,j) + lmncoord(i,a)*dNdxi(a,j);
          end
        end
      end

      dxidx = inv(dxdxi);
%
%     Convert shape function derivatives:derivatives wrt global coords
%
      for a = 1:n
        for i = 1:ncoord
          dNdx(a,i) = 0.;
          for j = 1:ncoord
            dNdx(a,i) = dNdx(a,i) + dNdxi(a,j)*dxidx(j,i);
          end
        end
      end
%
%     Compute the (infinitesimal) strain by differentiating displacements
%
      for i = 1:ncoord
         for j = 1:ncoord
            strain(i,j) = 0.;
            for a = 1:n
              strain(i,j) = strain(i,j) + 0.5*(displacements(i,a)*dNdx(a,j)+displacements(j,a)*dNdx(a,i));
            end
         end
      end

      if (ncoord == 2) 

      fprintf(outfile,'%5d %7.4f %7.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f \n', ...
        intpt,x(1),x(2),eplas(intpt,lmn),strain(1,1),strain(2,2),strain(1,2),stress(1,1,intpt,lmn),stress(2,2,intpt,lmn),stress(1,2,intpt,lmn));


      elseif (ncoord == 3) 

      fprintf(outfile,'%5d %7.4f %7.4f %7.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f \n',...
              intpt,x(1),x(2),x(3), eplas(intpt,lmn),...
              strain(1,1),strain(2,2),strain(3,3),strain(1,2),strain(1,3),strain(2,3), ...
              stress(1,1,intpt,lmn),stress(2,2,intpt,lmn),stress(3,3,intpt,lmn),stress(1,2,intpt,lmn),stress(1,3,intpt,lmn),stress(2,3,intpt,lmn));
      end
     end
   end
end

%==============================Function read_input_file =================
 function [nprops,materialprops,ncoord,ndof,nnode,coords,nelem,maxnodes, ...
                connect,nelnodes,elident,nfix,fixnodes,ndload,dloads] = read_input_file(infile) 

 cellarray=textscan(infile,'%s');

%  Total no. material parameters, && list of parameters
%
   nprops = str2num(cellarray{1}{2});
   materialprops = zeros(nprops,1);
   cellno = 2;
   for i = 1:nprops
     cellno = cellno + 2;  
     materialprops(i) = str2num(cellarray{1}{cellno});
   end;
%
%    no. coords (1:3), no. DOF, no. nodes && nodal coordinates
%
   cellno = cellno + 2;
   ncoord = str2num(cellarray{1}{cellno});
   cellno = cellno + 2;
   ndof = str2num(cellarray{1}{cellno});
   cellno = cellno + 2;
   nnode = str2num(cellarray{1}{cellno});
 
   coords = zeros(ncoord,nnode);
   cellno = cellno + 1;
   for i = 1 : nnode
     for j = 1 : ncoord
       cellno = cellno + 1;
       coords(j,i) = str2num(cellarray{1}{cellno});
     end;
   end;
%
%    No. elements && connectivity
%
   cellno = cellno + 2;
   nelem = str2num(cellarray{1}{cellno});
   cellno = cellno + 2;
   maxnodes = str2num(cellarray{1}{cellno});
   connect = zeros(maxnodes,nelem);
   nelnodes = zeros(nelem,1);
   elident = zeros(nelem,1);
   cellno = cellno + 3;
   for i = 1 : nelem
     cellno = cellno + 1;
     elident(i) = str2num(cellarray{1}{cellno});
     cellno = cellno + 1;
     nelnodes(i) = str2num(cellarray{1}{cellno});
     for j = 1 : nelnodes(i)
       cellno = cellno + 1;
       connect(j,i) = str2num(cellarray{1}{cellno});
     end
   end
%
%    No. nodes with prescribed displacements, with the prescribed displacements
% 
   cellno = cellno + 2;
   nfix = str2num(cellarray{1}{cellno});
   cellno = cellno + 3;
   fixnodes = zeros(3,nfix);
   for i = 1 : nfix
      cellno = cellno + 1;
      fixnodes(1,i) = str2num(cellarray{1}{cellno});
      cellno = cellno + 1;
      fixnodes(2,i) = str2num(cellarray{1}{cellno});
      cellno = cellno + 1;
      fixnodes(3,i) = str2num(cellarray{1}{cellno});
   end;
%
%    No. loaded element faces, with the loads
%
    cellno = cellno + 2;
    ndload = str2num(cellarray{1}{cellno});
    cellno = cellno + 3;
    dloads = zeros(2+ndof,ndload);
    for i = 1 : ndload
       cellno = cellno + 1;
       dloads(1,i) = str2num(cellarray{1}{cellno});
       cellno = cellno + 1;
       dloads(2,i) = str2num(cellarray{1}{cellno});
       for j = 1 : ndof
         cellno = cellno + 1;
         dloads(j+2,i) = str2num(cellarray{1}{cellno});
       end;
    end;
    
 end
  function plotmesh(coords,ncoord,nnode,connect,nelem,elident,nelnodes,color)
% Function to plot a mesh.  
    f2D_3 = [1,2,3];
    f2D_4 = [1,2,3,4];
    f2D_6 = [1,4,2,5,3,6];
    f2D_8 = [1,5,2,6,3,7,4,8];
    f3D_4 = [[1,2,3];[1,4,2];[2,4,3];[3,4,1]];
    f3D_10 = [[1,5,2,6,3,7];[1,8,4,9,2,5];[2,9,4,10,3,6];[3,10,4,8,1,7]];
    f3D_8 = [[1,2,3,4];[5,8,7,6];[1,5,6,2];[2,3,7,6];[3,7,8,4];[4,8,5,1]];
    f3D_20 = [[1,9,2,10,3,11,4,12];[5,16,8,15,7,14,6,13];
              [1,17,5,13,6,18,2,9];[2,18,6,14,7,19,3,10];
              [3,19,7,15,8,20,4,11];[4,20,8,16,5,17,1,12]];

   hold on
   if (ncoord==2)  % Plot a 2D mesh
       for lmn = 1:nelem
           for i = 1:nelnodes(lmn)
               x(i,1:2) = coords(1:2,connect(i,lmn));
           end
           scatter(x(:,1),x(:,2),'MarkerFaceColor','r');
           if (nelnodes(lmn)==3) 
               patch('Vertices',x,'Faces',f2D_3,'FaceColor','none','EdgeColor',color);
           elseif (nelnodes(lmn)==4)
               patch('Vertices',x,'Faces',f2D_4,'FaceColor','none','EdgeColor',color);
           elseif (nelnodes(lmn)==6) 
               patch('Vertices',x,'Faces',f2D_6,'FaceColor','none','EdgeColor',color);
           elseif (nelnodes(lmn)==8 || nelnodes(lmn)==9)
               patch('Vertices',x,'Faces',f2D_8,'FaceColor','none','EdgeColor',color);
           end
       end
   elseif (ncoord==3) % Plot a 3D mesh
       for lmn = 1:nelem
           for i = 1:nelnodes(lmn)
               x(i,1:3) = coords(1:3,connect(i,lmn));
           end
           scatter3(x(:,1),x(:,2),x(:,3),'MarkerFaceColor','r');
           if (nelnodes(lmn)==4) 
               patch('Vertices',x,'Faces',f3D_4,'FaceColor','none','EdgeColor',color);
           elseif (nelnodes(lmn)==10)
               patch('Vertices',x,'Faces',f3D_10,'FaceColor','none','EdgeColor',color);
           elseif (nelnodes(lmn)==8) 
               patch('Vertices',x,'Faces',f3D_8,'FaceColor','none','EdgeColor',color);
           elseif (nelnodes(lmn)==20)
               patch('Vertices',x,'Faces',f3D_20,'FaceColor','none','EdgeColor',color);
           end
       end    
   end
   axis equal
   hold off
end
 