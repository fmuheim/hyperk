#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<string>
#include <cmath>
#include "TMath.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TSystem.h"
#include "TH1F.h"
#include "TGraph.h"

//Reads the file
double stream (ifstream & stream)
{
    double tmpVal;
    stream >> tmpVal;
    return tmpVal;
}

void makeRootFile()
{
    gROOT->ProcessLine(".x lhcbStyle.C");
    gStyle->SetOptStat(110);

    TNtuple *myTupleOff = new TNtuple ("myTupleOff", "tup", "valueOff");

    //Read data 
    ifstream inOff;

    //Open coincidence and anticoincidence data files
    inOff.open("1hourAnticoincidence.Spe");

    //To remove metadata at top of files
    int j = 1;
    char line1[256];
    while (j < 13)
    {
        cout << j << endl;
        inOff.getline(line1, 256);
        ++j;
    }

    int ok = true;
    int valOff = 0;
    Double_t expected[2048];

    while (ok)
    {
        int val2 = stream(inOff);

        if (inOff.eof () == true || inOff.good () == false)
        {
            ok = false;
        }
        else
        {
            //Fill histograms
            for (int i = 0; i < val2; ++i)
            {   
                myTupleOff->Fill(valOff);
            }
            ++valOff;
        }
    }

    //TString tupleOn =TString::Format ("coincidence_on/fit/%.1fVoltsOnMyTuple.root", volts);
    TFile * file = TFile::Open("tuple.root", "RECREATE");
    myTupleOff->Write();
    file->Close();

    inOff.close();

    /*
    TCanvas *can2 =new TCanvas ("can2", "Histogram Anticoincidence", 200, 10, 800, 600);
    histoOff->Draw("HISTO");
    histoOff->Draw("SAME");
    gPad->SetLogy();
    histoOff->SetTitle("PMT Single Photon Spectrum Gate Off");
    histoOff->SetXTitle("ADC value");
    histoOff->SetYTitle("Count");
    histoOff->SetLineColor(kBlack);
    */
    }
