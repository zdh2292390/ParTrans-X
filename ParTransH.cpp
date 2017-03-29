//
//  TransH.cpp
//
//  Created by 张登辉 on 17/2/21.
//  Copyright (c) 2017年 张登辉. All rights reserved.
//
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<omp.h>
#include<algorithm>
using namespace std;


#define pi 3.1415926535897932384626433832795

string Int_to_String(int n)
{
    ostringstream stream;
    stream<<n;  //n为int类型
    return stream.str();
}
int String_to_Int(string s)
{
    stringstream ss;
    ss<<s;
    int num;
    ss>>num;
    return num;
}
double String_to_Double(string s)
{
    stringstream ss;
    ss<<s;
    double num;
    ss>>num;
    return num;
}
//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
    double res=0;
    for (int i=0; i<a.size(); i++)
        res+=a[i]*a[i];
    return sqrt(res);
}

bool cmp (const vector<int> a, const vector<int> b)
{
    return a[1] < b[1];
}

int rand_r_interval(int start, int end,unsigned int *seed)
{
    // int res = start+(rand_r(seed)*rand_r(seed))%(end-start+1);
    int res = start+(rand_r(seed))%(end-start+1);
    // while (res<0)
    //     res+=x;
    if(res>end){
        cout<<"大于end error"<<endl;
        exit(-1);
    }

    if(res<0){
        cout<<"小于0 error"<<endl;
        exit(-1);
    }
    return res;
}

string version;
string dataset="";
string resultpath="";
int nthread = 1;
int nepoch = 1000;
string strategy = "";

char buf[100000],buf1[100000];
int relation_num,entity_num,feature_num,nyt_relation_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<pair<int,int>, map<int,int> > ok;

map<int,map<int,vector<int> > > left_entity,right_entity;
map<int,double> left_mean,right_mean,left_var,right_var;
map<int,map<int,int > > left_candidate_ok,right_candidate_ok;
map<int,vector<int> > left_candidate,right_candidate;
map<int,int> entity2num;

vector<vector<int> > triples;
vector<int> splits;

class Train{
    
public:
    
    void add(int x,int y,int z)
    {
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
        ok[make_pair(x,z)][y]=1;
        vector<int> a;
        a.push_back(x);
        a.push_back(z);
        a.push_back(y);
        triples.push_back(a);
    }
    

    void run(int n_in,double rate_in,double margin_in,int method_in)
    {
        n = n_in;
        rate = rate_in;
        margin = margin_in;
        method = method_in;
        A.resize(relation_num);
        for (int i=0; i<relation_num; i++)
        {
            A[i].resize(n);
            for (int j=0; j<n; j++)
                A[i][j] = randn(0,1.0/n,-1,1);
            norm2one(A[i]);
        }
        relation_vec.resize(relation_num);
        for (int i=0; i<relation_vec.size(); i++)
            relation_vec[i].resize(n);
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_vec.size(); i++)
            entity_vec[i].resize(n);
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-1,1);
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-1,1);
        }
        
        bfgs();
    }
    
private:
    int n;
    int method;
    double res;//loss function value
    double count,count1;//loss function gradient
    double rate;//learning rate
    double belta;
    double margin;
    vector<int> fb_h,fb_l,fb_r;
    vector<vector<int> > feature;
    vector<vector<double> > A, A_tmp;
    vector<vector<double> > relation_vec,entity_vec,relation_tmp,entity_tmp;
    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
            for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }
    double norm2one(vector<double> &a)
    {
        double x = vec_len(a);
        for (int ii=0; ii<a.size(); ii++)
            a[ii]/=x;
        return 0;
    }
    double norm(vector<double> &a, vector<double> &A)
    {
        norm2one(A);
        double sum=0;
        while (true)
        {
            for (int i=0; i<n; i++)
                sum+=sqr(A[i]);
            sum = sqrt(sum);
            for (int i=0; i<n; i++)
                A[i]/=sum;
            double x=0;
            for (int ii=0; ii<n; ii++)
            {
                x+=A[ii]*a[ii];
            }
            if (x>0.1)
            {
                for (int ii=0; ii<n; ii++)
                {
                    a[ii]-=rate*A[ii];
                    A[ii]-=rate*a[ii];
                }
            }
            else
                break;
        }
        norm2one(A);
        return 0;
    }
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    int rand_r_max(int x,unsigned int *seed )
    {
        int res = (rand_r(seed)*rand_r(seed))%x;
        while (res<0)
            res+=x;
        return res;
    }

    
    void bfgs()
    {
        int step=0;
        count=0;
        count1=0;
        res=0;
        double times=0;
        int nbatches=50;
        // int neval = 1000;
        int batchsize = fb_h.size()/nbatches;
        
        // A_tmp = A;
        // relation_tmp = relation_vec;
        // entity_tmp = entity_vec;

        int interval=fb_h.size()/nthread;
        int npartition=splits.size()/nthread;

        #pragma omp parallel for firstprivate(ok)
        for(int t=0;t<nthread;t++)
        {   
            time_t start,stop;
            double private_res;
            srand((unsigned) time(NULL)+t);//每个线程设置不同的采样种子，防止采样冲突
            unsigned int seed = (unsigned) time(NULL)+t;

            for (int eval=0; eval<nepoch/nthread; eval++)
            {
                res=0;
                private_res=0;
                start = time(NULL);
                for (int batch = 0; batch<nbatches; batch++)
                {
                    for (int k=0; k<batchsize; k++)
                    {
                        
                        int i;
                        if(strategy=="TransH")
                            i=rand_r_max(fb_h.size(),&seed);
                        else if(strategy=="TransH_split")
                        {
                            int m=t%splits.size();

                            if(m!=splits.size()-1)
                                i=rand_r_interval(splits[m],splits[m+1]-1,&seed);
                            else
                                i=rand_r_interval(splits[m],fb_h.size()-1,&seed);

                            if(i>=triples.size()||i<0)
                            {
                                cout<<i<<endl;
                                cout<<"error"<<endl;
                                exit(-1);
                            }
                        }
                        else
                        {
                            printf("strategy error!\n");
                            exit(-1);
                        }

                        int j=rand_r_max(entity_num,&seed);//随机找反例
                        // double pr = 1000*right_mean[fb_r[i]]/(right_mean[fb_r[i]]+left_mean[fb_r[i]]);
                        double pr=500;
                        if (method ==0)
                            pr = 500;

                        int flag = (rand_r(&seed)%1000<pr);

                        if (flag)
                        {
                            while(true)
                            {
                                if(!(ok.find(make_pair(fb_h[i],fb_r[i]))!=ok.end() && ok[make_pair(fb_h[i],fb_r[i])].count(j)>0))
                                    break;
                                j=rand_r_max(entity_num,&seed);
                            }
                            train_kb(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],private_res);
                        }
                        else
                        {
                            while(true)
                            {
                                if(!(ok.find(make_pair(j,fb_r[i]))!=ok.end() && ok[make_pair(j,fb_r[i])].count(fb_l[i])>0))
                                    break;
                                j=rand_r_max(entity_num,&seed);
                            }

                            train_kb(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],private_res);
                        }
                        // norm(entity_tmp[fb_h[i]]);
                        // norm(entity_tmp[fb_l[i]]);
                        // norm(entity_tmp[j]);
                        // norm(entity_tmp[fb_h[i]],A_tmp[fb_r[i]]);
                        // norm(entity_tmp[fb_l[i]],A_tmp[fb_r[i]]);
                        // norm(entity_tmp[j],A_tmp[fb_r[i]]);

                        // norm(entity_vec[fb_h[i]]);
                        // norm(entity_vec[fb_l[i]]);
                        // norm(entity_vec[j]);
                        // norm(entity_vec[fb_h[i]],A[fb_r[i]]);
                        // norm(entity_vec[fb_l[i]],A[fb_r[i]]);
                        // norm(entity_vec[j],A[fb_r[i]]);

                    }
               	}
                stop = time(NULL);
                printf("thread id:%d,epoch: %d,res: %f, time:%d\n", omp_get_thread_num(),eval,private_res,(stop-start));

            }
        }

        FILE* f1 = fopen((resultpath+"/A."+version).c_str(),"w");
        FILE* f2 = fopen((resultpath+"/relation2vec."+version).c_str(),"w");
        FILE* f3 = fopen((resultpath+"/entity2vec."+version).c_str(),"w");
        
        for (int i=0; i<relation_num; i++)
        {
            for (int jj=0; jj<n; jj++)
                fprintf(f1,"%.6lf\t",A[i][jj]);
            fprintf(f1,"\n");
        }

        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
            fprintf(f2,"\n");
        }
        
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
            fprintf(f3,"\n");
        }
        fclose(f1);
        fclose(f2);
        fclose(f3);

    }
    double res1;
    double calc_sum(int e1,int e2,int rel)
    {
        double tmp1=0,tmp2=0;
        for (int jj=0; jj<n; jj++)
        {
            tmp1+=A[rel][jj]*entity_vec[e1][jj];
            tmp2+=A[rel][jj]*entity_vec[e2][jj];
        }
        
        double sum=0;
        for (int ii=0; ii<n; ii++)
            sum+=fabs(entity_vec[e2][ii]-tmp2*A[rel][ii]-(entity_vec[e1][ii]-tmp1*A[rel][ii])-relation_vec[rel][ii]);
        return sum;
    }
    void gradient(int e1,int e2,int rel,double belta)
    {
        double tmp1 = 0, tmp2 = 0;
        double sum_x=0;
        for (int jj=0; jj<n; jj++)
        {
            tmp1+=A[rel][jj]*entity_vec[e1][jj];
            tmp2+=A[rel][jj]*entity_vec[e2][jj];
        }
        for (int ii=0; ii<n; ii++)
        {
            
            double x = 2*(entity_vec[e2][ii]-tmp2*A[rel][ii]-(entity_vec[e1][ii]-tmp1*A[rel][ii])-relation_vec[rel][ii]);
            //for L1 distance function
            if (x>0)
                x=1;
            else
                x=-1;
            sum_x+=x*A[rel][ii];
            // relation_tmp[rel][ii]-=belta*rate*x;
            // entity_tmp[e1][ii]-=belta*rate*x;
            // entity_tmp[e2][ii]+=belta*rate*x;
            // A_tmp[rel][ii]+=belta*rate*x*tmp1;
            // A_tmp[rel][ii]-=belta*rate*x*tmp2;

            relation_vec[rel][ii]-=belta*rate*x;
            entity_vec[e1][ii]-=belta*rate*x;
            entity_vec[e2][ii]+=belta*rate*x;
            A[rel][ii]+=belta*rate*x*tmp1;
            A[rel][ii]-=belta*rate*x*tmp2;

        }
        for (int ii=0; ii<n; ii++)
        {
            // A_tmp[rel][ii]+=belta*rate*sum_x*entity_vec[e1][ii];
            // A_tmp[rel][ii]-=belta*rate*sum_x*entity_vec[e2][ii];

            A[rel][ii]+=belta*rate*sum_x*entity_vec[e1][ii];
            A[rel][ii]-=belta*rate*sum_x*entity_vec[e2][ii];
        }
        
        // norm(relation_tmp[rel]);
        // norm(entity_tmp[e1]);
        // norm(entity_tmp[e2]);
        // norm2one(A_tmp[rel]);
        // norm(relation_tmp[rel],A_tmp[rel]);

        norm(relation_vec[rel]);
        norm(entity_vec[e1]);
        norm(entity_vec[e2]);
        norm2one(A[rel]);
        norm(relation_vec[rel],A[rel]);
    }
    void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,double &private_res)
    {
        double sum1 = calc_sum(e1_a,e2_a,rel_a);
        double sum2 = calc_sum(e1_b,e2_b,rel_b);
        if (sum1+margin>sum2)
        {
            private_res+=margin+sum1-sum2;
            gradient( e1_a, e2_a, rel_a, -1);
            gradient(e1_b, e2_b, rel_b,1);
        }
    }
};

void split()
{   
    splits.push_back(0);
    for (int i = 1; i < triples.size(); ++i)
    {
        if(triples[i][1]!=triples[i-1][1])
            splits.push_back(i);
    }

}

Train train;
void prepare()
{

    FILE* f1 = fopen(("./"+dataset+"/entity2id.txt").c_str(),"r");
    FILE* f2 = fopen(("./"+dataset+"/relation2id.txt").c_str(),"r");
    if(!f1)
    {
        printf(("can't open ./"+dataset+"/entity2id.txt").c_str());
        exit(-1);
    } 
    if(!f2)
    {
        printf(("can't open ./"+dataset+"/relation2id.txt").c_str());
        exit(-1);  
    } 
    int x;
    while (fscanf(f1,"%s%d",buf,&x)==2)
    {
        string st=buf;
        entity2id[st]=x;
        id2entity[x]=st;
        entity_num++;
    }
    while (fscanf(f2,"%s%d",buf,&x)==2)
    {
        string st=buf;
        relation2id[st]=x;
        id2relation[x]=st;
        relation_num++;
    }

    FILE* f_kb = fopen(("./"+dataset+"/train.txt").c_str(),"r");
    if(!f_kb)
    {
        printf(("can't open ./"+dataset+"/train.txt").c_str());
        exit(-1);
    }

    while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel = relation2id[s3];
        if (left_candidate_ok[rel].count(e1)==0)
        {
            left_candidate_ok[rel][e1]=1;
            left_candidate[rel].push_back(e1);
        }
        if (right_candidate_ok[rel].count(e2)==0)
        {
            right_candidate_ok[rel][e2]=1;
            right_candidate[rel].push_back(e2);
        }
        entity2num[e1]++;
        entity2num[e2]++;
        left_entity[rel][e1].push_back(e2);
        right_entity[rel][e2].push_back(e1);
        train.add(e1,e2,rel);
    }
    for (int i=0; i<relation_num; i++)
    {
        double sum1=0,sum2=0,sum3 = 0;
        for (map<int,vector<int> >::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
        {
            sum1++;
            sum2+=it->second.size();
            sum3+=sqr(it->second.size());
        }
        left_mean[i]=sum2/sum1;
        
        left_var[i]=sum3/sum1-sqr(left_mean[i]);
    }
    for (int i=0; i<relation_num; i++)
    {
        double sum1=0,sum2=0,sum3=0;
        for (map<int,vector<int> >::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
        {
            sum1++;
            sum2+=it->second.size();
            sum3+=sqr(it->second.size());
        }
        right_mean[i]=sum2/sum1;
        right_var[i]=sum3/sum1-sqr(right_mean[i]);
    }
    
    // for (int i=0; i<relation_num; i++)
    //     cout<<i<<'\t'<<id2relation[i]<<' '<<left_mean[i]<<' '<<right_mean[i]<<endl;
    
    fclose(f_kb);
    sort(triples.begin(),triples.end(),cmp);
    split();
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}


int main(int argc,char**argv)
{
   srand((unsigned) time(NULL));
    int n = 100;
    double rate = 0.001;
    double margin = 1;
    int method = 0;
    int i;
    
    if (method)
        version = "bern";
    else
        version = "unif";
    

    if (argc<2)
    {
        printf("not enough Argument\n" );
    }
    else
    {
        version = argv[1];
        nthread = String_to_Int(argv[2]);
        dataset = argv[3];
        n=String_to_Int(argv[4]);
        margin=String_to_Double(argv[5]);
        nepoch=String_to_Int(argv[6]);
        resultpath=argv[7];
        rate= String_to_Double(argv[8]);
        strategy = argv[9];
       
        cout<<"strategy:"<<strategy<<" dataset:"<<dataset<<" nthread:"<<nthread<<" k:"<<n<<" margin:"<<margin<<" nepoch:"<<nepoch<<" rate:"<<rate<<" version:"<<version<<endl;

        time_t start,stop;
        start = time(NULL);
        prepare();
        stop = time(NULL);
        printf("Prepare Time:%ld\n",(stop-start));

        start=time(NULL);
        train.run(n,rate,margin,method);
        stop = time(NULL);

        int traintime = stop-start;
        printf("Train Time:%ld\n",(stop-start));

        FILE* f = fopen("./all.TransH.log","a");
        fprintf(f,"strategy:%s dataset:%s version:%s nthread:%d k:%d margin:%f nepoch:%d rate:%f \n",strategy.c_str(),dataset.c_str(),version.c_str(),nthread,n,margin,nepoch,rate);
        fprintf(f, "traintime:%d\n", traintime);
        // fprintf(f,"\n");
        fclose(f);
    }
    
}



