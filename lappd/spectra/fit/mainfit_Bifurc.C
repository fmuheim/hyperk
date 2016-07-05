#ifdef __CINT__
#define ROOT_GuiTypes 0
#endif

#include <TStyle.h>
#include <TCanvas.h>
#include "TH1.h"
#include "TROOT.h"
#include "TRandom.h"
#include "TAxis.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "TBrowser.h"
#include "TMath.h"
#include "TF1.h"

using namespace std;

//Scale factors
//Double_t scale = 40.0;        //old default is 20.0
//Double_t scale_x0 = 40.0;     //old default is 20.0
//Double_t scale_sigma = 30.0;  //old default is 15.0
Double_t XTalkGainRatio = 0.05; //default is 0.05

Double_t finalFunc(Double_t *x, Double_t *par)
{
  Double_t sigma, x0, total, ratio, poiss, gain, gain_xtalk, diff, sigma_norm;
  Int_t index, signalindex, signalindexnumber;
  Int_t kindex, signalkindex, signalkindexnumber;
  Double_t arean=0., areak=0.,  area=0., areatotal=0.;
  Int_t n=3;
  gain = par[3]/par[6];
  gain_xtalk = gain * XTalkGainRatio;
  total = 0;
 
  //signal loop (for index from 0 to 3)
  for(index = 0; index <= n; index ++){

    if (index == 0){ arean = par[0] * TMath::Exp( -par[4]); }
    else{ arean = par[0] * TMath::Poisson(index , par[4] ); }

    // x-talk loop (for kindex from 0 to 2) 
    for(kindex = 0; kindex <= 2; kindex ++){ 
      if (kindex == 0){ areak = TMath::Exp( -par[5]); } 
      else { areak = TMath::Poisson( kindex, par[5]); }
      area = arean * areak;

      if (index == 0 && kindex == 0){ //i.e. the pedestal
	x0 = par[1]; 
        diff =  x[0] - x0;
        sigma_norm = 0.5 * (par[2] + par[8]);
        if ( diff > 0) sigma = par[8];
        else sigma = par[2];
	total = total + area * TMath::Exp(-0.5 * (x[0]-x0) * (x[0]-x0)/ sigma / sigma)/(2.50663 * sigma_norm);}

      else{
	signalindexnumber = (index + kindex) * 10 + 10;
	for(signalindex = 0; signalindex <= signalindexnumber; signalindex ++){
	  ratio = index * gain + kindex * gain_xtalk;
	  if (signalindex == 0){
	    poiss = TMath::Exp( -ratio); }
	  else{
	    poiss = TMath::Poisson(signalindex , ratio); }
	  areatotal = poiss * area;
	  sigma = TMath::Sqrt(par[2]*par[2] + signalindex * par[7] * par[7]);
	  x0 = par[1] + signalindex * par[6];
	  total = total + areatotal * TMath::Exp(-0.5 * (x[0]-x0) * (x[0]-x0)/ sigma / sigma)/(2.50663 * sigma);}
      } //end of else
    } //end of for kindex loop 
  } //end of for index loop

  return total;
}

//sgl Gauss
//Double_t plotpe( Double_t *x, Double_t *par) 
//{ return par[0] * TMath::Exp( -par[3]) * TMath::Exp( -par[4]) * TMath::Exp( -0.5 * (x[0] - par[1]) * ( x[0] - par[1])/ par[2] / par[2]) / (2.50663 * par[2]) ;}

//bifurc Gauss
Double_t plotpe( Double_t *x, Double_t *par) 
{
  Double_t sigma, diff, sigma_norm;
  diff= x[0] - par[1];
  sigma_norm = 0.5 * (par[2] + par[5]);
  if ( diff > 0) sigma = par[5];
  else sigma = par[2];
 return par[0] * TMath::Exp( -par[3]) * TMath::Exp( -par[4]) * TMath::Exp( -0.5 * (x[0] - par[1]) * ( x[0] - par[1])/ sigma / sigma) / (2.50663 * sigma_norm) ;
}


Double_t plotsi1( Double_t *x, Double_t *par)
{
  Double_t sigma, x0, total, areatotal, poiss1, gain, ratio, area;
  Int_t signalindex;
  area = par[0] * TMath::Poisson(1, par[4]) * TMath::Exp( -par[5]);
  gain = par[3]/par[6];
  total = 0;

  for(signalindex = 0; signalindex <= 20; signalindex ++){ 
    ratio = gain;
    if (signalindex == 0){
      poiss1 = TMath::Exp( -ratio);}
    else{
      poiss1 = TMath::Poisson(signalindex , ratio);
    }
    areatotal = poiss1 * area;
    sigma = TMath::Sqrt(par[2]*par[2] + signalindex * par[7] * par[7]);
    x0 = par[1] + signalindex * par[6];
    total = total + areatotal * TMath::Exp(-0.5 * (x[0]-x0) * (x[0]-x0)/ sigma / sigma)/(2.50663 * sigma);}
  return total;
}


Double_t plotsi2( Double_t *x, Double_t *par)
{
  Double_t sigma, x0, total, areatotal, poiss1, gain, ratio, area;
  Int_t signalindex;
  area = par[0] * TMath::Poisson(2, par[4]) * TMath::Exp( -par[5]);
  gain = par[3]/par[6];
  total = 0;

  for(signalindex = 0; signalindex <= 30; signalindex ++){ 
    ratio = 2 * gain;
    if (signalindex == 0){
      poiss1 = TMath::Exp( -ratio);}
    else{
      poiss1 = TMath::Poisson(signalindex , ratio);
    }
    areatotal = poiss1 * area;
    sigma = TMath::Sqrt(par[2]*par[2] + signalindex * par[7] * par[7]);
    x0 = par[1] + signalindex * par[6];
    total = total + areatotal * TMath::Exp(-0.5 * (x[0]-x0) * (x[0]-x0)/ sigma / sigma)/(2.50663 * sigma);}
  return total;
}

Double_t plotsi3( Double_t *x, Double_t *par)
{
  Double_t sigma, x0, total, areatotal, poiss1, gain, ratio, area;
  Int_t signalindex;
  area = par[0] * TMath::Poisson(3, par[4]) * TMath::Exp( -par[5]);
  gain = par[3]/par[6];
  total = 0;

  for(signalindex = 0; signalindex <= 40; signalindex ++){
    ratio = 3 * gain;
    if (signalindex == 0){
      poiss1 = TMath::Exp( -ratio);}
    else{
      poiss1 = TMath::Poisson(signalindex , ratio);
    }
    areatotal = poiss1 * area;
    sigma = TMath::Sqrt(par[2]*par[2] + signalindex * par[7] * par[7]);
    x0 = par[1] + signalindex * par[6];
    total = total + areatotal * TMath::Exp(-0.5 * (x[0]-x0) * (x[0]-x0)/ sigma / sigma)/(2.50663 * sigma);}
  return total;
}


Double_t plotsignalplusxtalk( Double_t *x, Double_t *par)  //This is 1 signal p.e.  +  1 Xtalk p.e.
{
  Double_t sigma, x0, total, areatotal, poiss, gain, ratio, area, gain_xtalk;
  Int_t signalindex;
  area = par[0] * TMath::Poisson(1, par[4]) * TMath::Poisson(1, par[6]);
  gain = par[3]/par[7];
  gain_xtalk = par[5]/par[7];
  total = 0;
  
  for(signalindex = 0; signalindex <= 30; signalindex ++){ 
	  ratio = gain + gain_xtalk;
	  if (signalindex == 0){
	    poiss = TMath::Exp( -ratio); }
	  else{
	    poiss = TMath::Poisson(signalindex , ratio);}
	  areatotal = poiss * area;
	  sigma = TMath::Sqrt(par[2]*par[2] + signalindex * par[8] * par[8]);
	  x0 = par[1] + signalindex * par[7];
	  total = total + areatotal * TMath::Exp(-0.5 * (x[0]-x0) * (x[0]-x0)/ sigma / sigma)/(2.50663 * sigma);}

  return total;
}



// ------ main function -----------------------
// --------------------------------------------
// --------------------------------------------
void mainfit_Bifurc(int pixelNumber)
{   
  //int pixelNumber = 0; 
  std::string address, scanfolder, inputname;
  //address = "/Disk/speyside4/lhcb/mapmt/runs/";  //This is always the same
  address = "inputs/";
  //scanfolder = address + "scan-newpmt-bl-rt-900/";  //This can change
  //scanfolder = address + "scan-newpmt-bl-rt-1000/";
  //scanfolder = address + "FlatPanel/";
  scanfolder = address + "gainMap/";
  char filename[100], savename1[100], savename2[100];
  cout<<"The address is: "<<scanfolder<<endl;
  stringstream ss;
  double k, value, position, posnum, position1;
  double N, Nerr, p1, p1err, p2, p2err, p3, p3err, p4, p4err, p5, p5err, p6, p6err, p7, p7err, p8, p8err;
  double p0low, p1low, p2low, p3low, p4low, p5low, p6low, p7low, p8low; 
  double p0seed, p1seed, p2seed, p3seed, p4seed, p5seed, p6seed, p7seed, p8seed; 
  double p0high, p1high, p2high, p3high, p4high, p5high, p6high, p7high, p8high;
  int channel, t, num, jk, Nbins, xmax;

  Nbins = 200;  // should be 200 for 1000V, and 100 for 900V
  xmax = 2000;  // should be 2000 for 1000V, and 1000 for 900V
  TH1F *h = new TH1F("h","MaPMT histogram",Nbins,0,xmax);
  TH1F *h0 = new TH1F("h0","MaPMT histogram0",Nbins,0,xmax);
  //inputname = scanfolder + "fwd_0.08.root";
  //inputname = scanfolder + "20140819_3.6V.root"; //FlatPanel, 1000V
  //inputname = scanfolder + "20140819_3.6V_900V.root"; //FlatPanel, 900V
  //inputname = scanfolder + "FA0002_bl.root"; //gainMap, high gain
  inputname = scanfolder + "ZN0971_bl.root"; //gainMap, low gain
  //inputname = scanfolder + "FA0006_bl.root"; //gainMap, low gain
  cout<<"The input file is: "<<inputname<<endl;

  TFile* inputFile = new TFile(inputname.c_str());
  if (inputFile->IsZombie()){
      std::cout << "Failed to find input file" << std::endl;
      return;
      }
  TTree* tree = (TTree*)inputFile->Get("QDCReadoutTree");
  if (!tree){
      std::cout << "Failed to find input tree" << std::endl;
      return;
      }
  //tree->Draw("Value[0]");
		
  //Put values into histogram
  int raw[32];
  tree->SetBranchAddress("Value",raw);
  int nevent = tree->GetEntries();
  for (int ievent =0 ; ievent<nevent; ++ievent){
           tree->GetEntry(ievent);
           k = raw[pixelNumber] * 0.1; //scaling by bin width
           h->AddBinContent(k+1);  
           }

  cout << "End-of-file reached.." << endl;
  TCanvas *cw= new TCanvas("cw","MaPMTs Test",200,10,650,500);
  //TCanvas *cw0= new TCanvas("cw0","MaPMTs Test0",200,10,650,500);
  h -> Draw();
  value = h -> GetMaximum();
  position = h -> GetMaximumBin();
  h0 = (TH1F * )h ->Clone();
  //this estimates the maximum signal value position, above the pedestal
  posnum = position + 25; //should be +10 for low-gain spectra, and +25 for high-gain spectra
  for ( int pos1 = 0; pos1 < posnum; pos1++ ) {
      	    h0 -> SetBinContent(pos1,0);
  }
                        
  position1 = h0 -> GetMaximumBin();
  position1 = position1 * 10; //factor 10 is from 200 bins in (0,2000)
  position = position * 10;
  position1 = position1 - position; //give a better estimate for the signal size
  cout<< "position = " << position << ", position1 = " << position1 <<endl;

  cout<< "XTalkGainRatio is " << XTalkGainRatio  <<endl;

  //Set seeds and limits
  p0low  = 300000;
  p0seed = 500000;               //integral of plot (Nevts * 10) 
  p0high = 700000;               

  p1low  = position - 20;
  p1seed = position;           // pedestal position
  p1high = position + 20;

  p2low  = 3.0;                 
  p2seed = 15.0;                // pedestal width (LHS)
  p2high = 30.0;

  p3low  = position1 - 520;    //was-120
  p3seed = position1;          // signal gain 
  p3high = position1 + 520;   //was + 120 

  p4low  = 0.01;            //was 0.1 
  p4seed = 0.5;                // average number of signal phe. per event
  p4high = 0.8;

  p5low  = 0.01;
  p5seed = 0.1;                // average number of XTalk phe. per event
  p5high = 0.5; 

  p6low  = 10.0;   //use 10, 20, 50 for low-gain spectra, and 10, 40, 100 for high-gain spectra
  p6seed = 20.0;               // scaling for gain (formerly known as scale and scale_x0)
  p6high = 50.0;  

  p7low  = 5.0;   //use 5, 15, 40 for low-gain spectra, and 10, 30, 100 for high-gain spectra
  p7seed = 15.0;                // scaling for width (formerly known as scale_sigma)
  p7high = 40.0; 

  p8low  = 5.0;                 
  p8seed = 30.0;                // pedestal width (RHS)
  p8high = 60.0;


  Double_t para[9];
  TF1 *pgaddfit1= new TF1("pgaddfit1",finalFunc,0,xmax,9);
  pgaddfit1->SetParameter(0,p0seed);
  pgaddfit1->SetParLimits(0,p0low,p0high);
  pgaddfit1->SetParameter(1,p1seed);        
  pgaddfit1->SetParLimits(1,p1low,p1high);
  pgaddfit1->SetParameter(2,p2seed);        
  pgaddfit1->SetParLimits(2,p2low,p2high);
  pgaddfit1->SetParameter(3,p3seed);       
  pgaddfit1->SetParLimits(3,p3low,p3high);
  pgaddfit1->SetParameter(4,p4seed);       
  pgaddfit1->SetParLimits(4,p4low,p4high);
  pgaddfit1->SetParameter(5,p5seed);       
  pgaddfit1->SetParLimits(5,p5low,p5high);
  pgaddfit1->SetParameter(6,p6seed);       
  pgaddfit1->SetParLimits(6,p6low,p6high);
  pgaddfit1->SetParameter(7,p7seed);       
  pgaddfit1->SetParLimits(7,p7low,p7high);
  pgaddfit1->SetParameter(8,p8seed);       
  pgaddfit1->SetParLimits(8,p8low,p8high);

  pgaddfit1->SetLineColor(kRed);
  pgaddfit1 -> SetNpx(20000);
  pgaddfit1 -> SetLineWidth(1);
  h->Fit("pgaddfit1","RN");
  pgaddfit1 -> GetParameters(para);

  h -> GetYaxis() -> SetTitle("Events number");
  h -> GetYaxis() -> CenterTitle();
  h -> GetYaxis() -> SetTitleOffset( 1.15);
  h -> GetYaxis() -> SetTitleSize( 0.04);
  h -> GetYaxis() -> SetRangeUser(0,500); //for linear y-axis
  //h -> GetXaxis() -> SetRangeUser(0,400); 
  //gPad -> SetLogy();
  h -> GetXaxis() -> SetTitle("x");
  h -> Draw();
			
  pgaddfit1 -> Draw("same");
			
  //TF1 *totalf = new TF1("totalf",finalFunc, 0, xmax, 6);
  //totalf -> SetParameters(para[0], para[1], para[2], para[3], para[4], para[5]);
  //totalf -> SetLineColor(kBlack);
  //totalf -> SetLineWidth(2);
  //totalf -> Draw("same");

  TF1 *pedesf = new TF1("pedesf",plotpe, 0, xmax, 6);
  pedesf -> SetParameters(para[0], para[1], para[2], para[4], para[5], para[8]);
  pedesf -> SetLineColor(kBlue);
  pedesf -> SetLineWidth(1);
  pedesf -> Draw("same");

  TF1 *signalf = new TF1("signalf", plotsi1, 0, xmax, 8);
  signalf -> SetParameters(para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7]);
  signalf -> SetLineColor(30); //dull green
  signalf -> SetLineWidth(2);
  signalf -> Draw("same");

  TF1 *signalf2 = new TF1("signalf2", plotsi2, 0, xmax, 8);
  signalf2 -> SetParameters(para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7]);
  signalf2 -> SetLineColor(50); //sandstone red
  signalf2 -> SetLineWidth(2);
  signalf2 -> Draw("same");

  TF1 *signalf3 = new TF1("signalf3", plotsi3, 0, xmax, 8);
  signalf3 -> SetParameters(para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7]);
  signalf3 -> SetLineColor(6); //bright pink
  signalf3 -> SetLineWidth(2);
  signalf3 -> Draw("same");

  Double_t XTalkGain = para[3] * XTalkGainRatio; //XTalk gain
  //Double_t XTalkGain = para[3] * 0.5; //for debugging XTalk shape
  TF1 *xtalkf = new TF1("xtalkf", plotsi1, 0, xmax, 8);
  xtalkf -> SetParameters(para[0], para[1], para[2], XTalkGain, para[5], para[4], para[6], para[7]);
  xtalkf -> SetLineColor(40); //grey
  xtalkf -> SetLineWidth(2);
  xtalkf -> Draw("same");
                  
  TF1 *xtalkf2 = new TF1("xtalkf2", plotsi2, 0, xmax, 8); //two XTalk p.e.
  xtalkf2 -> SetParameters(para[0], para[1], para[2], XTalkGain, para[5], para[4], para[6], para[7]);
  xtalkf2 -> SetLineColor(20); //beige
  xtalkf2 -> SetLineWidth(2);
  xtalkf2 -> Draw("same");
                        
  TF1 *signalplusxtalkf = new TF1("signalplusxtalkf", plotsignalplusxtalk, 0, xmax, 9); //1+1
  signalplusxtalkf -> SetParameters(para[0], para[1], para[2], para[3], para[4], XTalkGain, para[5], para[6], para[7]);
  signalplusxtalkf -> SetLineColor(7); //aquamarine
  signalplusxtalkf -> SetLineWidth(2);
  signalplusxtalkf -> Draw("same");


  TLegend *leg = new TLegend(0.58, 0.55, 0.9, 0.9);
  leg -> SetTextFont(30);
  leg -> SetTextSize(0.03);
  leg -> AddEntry(pgaddfit1, "global fit", "l");
  leg -> AddEntry(pedesf, "pedestal", "l");
  leg -> AddEntry(signalf, "one photoel", "l");
  leg -> AddEntry(signalf2, "two photoel", "l");
  leg -> AddEntry(signalf3, "three photoel", "l");
  leg -> AddEntry(xtalkf, "one Xtalk phe", "l");
  leg -> AddEntry(xtalkf2, "two Xtalk phe", "l");
  leg -> AddEntry(signalplusxtalkf, "one signal, one Xtalk", "l");
  leg -> Draw();

  h -> SetStats(0);
  //cw->Print("output.pdf");
  sprintf(savename1,"output_%d.pdf",pixelNumber);
  cw->Print(savename1);

  delete cw;
  delete h;
  delete h0;
  delete pgaddfit1;    
      
  }
