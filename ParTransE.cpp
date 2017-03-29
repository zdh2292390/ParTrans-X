#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<omp.h>
#include<sstream>
using namespace std;


#define pi 3.1415926535897932384626433832795

bool L1_flag=1;

string dataset="";
string resultpath="";
int nthread = 1;
int nepoch = 1000;
string strategy = "";
string version;

char buf[100000],buf1[100000];
int relation_num,entity_num;

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
vector<vector<double> > relation_vec,entity_vec;

map<int,map<int,int> > left_entity,right_entity;
map<int,double> left_num,right_num;
map<pair<int,int>, map<int,int> > ok;

string Int_to_String(int n)
{
    ostringstream stream;
    stream<<n;
    return stream.str();
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
    res = sqrt(res);
    return res;
}


class Train{
    
public:
    void add(int x,int y,int z)
    {
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
        ok[make_pair(x,z)][y]=1;
    }
    void run(int n_in,double rate_in,double margin_in,int method_in)
    {
        n = n_in;
        rate = rate_in;
        margin = margin_in;
        method = method_in;
        relation_vec.resize(relation_num);
        for (int i=0; i<relation_vec.size(); i++)
            relation_vec[i].resize(n);
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_vec.size(); i++)
            entity_vec[i].resize(n);
        relation_tmp.resize(relation_num);
        for (int i=0; i<relation_tmp.size(); i++)
            relation_tmp[i].resize(n);
        entity_tmp.resize(entity_num);
        for (int i=0; i<entity_tmp.size(); i++)
            entity_tmp[i].resize(n);
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]);
        }
        bfgs();
    }
    
private:
    int n,method;
    double rate,margin;

    double res;//loss function value
    double count,count1;//loss function gradient

    double belta;
    vector<int> fb_h,fb_l,fb_r;
    vector<vector<int> > feature;
    
    vector<vector<double> > relation_tmp,entity_tmp;
    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
            for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
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
        int nbatches=100;
        int batchsize = fb_h.size()/nbatches;

        #pragma omp parallel for firstprivate(ok)
        for(int t=0;t<nthread;t++)
        {   
            time_t start,stop;
            double private_res;
         

            unsigned int seed = (unsigned) time(NULL)+t;
            for (int epoch=0; epoch<nepoch/nthread; epoch++)
            {
                start = time(NULL);
                res=0;
                private_res=0;
            for (int batch = 0; batch<nbatches; batch++)
            {

                for (int k=0; k<batchsize; k++)
                {
                    
                    int i=rand_r_max(fb_h.size(),&seed);
                    int j=rand_r_max(entity_num,&seed);

                    if(i>=fb_h.size() || j>=entity_num)
                        cout<<"i:"<<i<<" j:"<<j<<endl;
                    double pr = 1000*right_num[fb_r[i]]/(right_num[fb_r[i]]+left_num[fb_r[i]]);
                    if (method ==0)//均匀分布
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
                    norm(relation_vec[fb_r[i]]);
                    norm(entity_vec[fb_h[i]]);
                    norm(entity_vec[fb_l[i]]);
                    norm(entity_vec[j]);
                }
            }
            stop = time(NULL);
            printf("thread id:%d,epoch: %d,res: %f, time:%d\n", omp_get_thread_num(),epoch,private_res,(stop-start));
            }
        }

        for (int i=0; i<entity_num;i++)
        {

            if (vec_len(entity_vec[i])-1>1e-3)
            {
                cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
                norm(entity_vec[i]);
            }
                
        }

        FILE* f2 = fopen((resultpath+"/relation2vec."+version).c_str(),"w");
        FILE* f3 = fopen((resultpath+"/entity2vec."+version).c_str(),"w");

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
        fclose(f2);
        fclose(f3);

    }
    double res1;
    double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
            for (int ii=0; ii<n; ii++)
                sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
            for (int ii=0; ii<n; ii++)
                sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);

        return sum;
    }
    void gradient(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
        for (int ii=0; ii<n; ii++)
        {
            double x = 2*(entity_vec[e2_a][ii]-entity_vec[e1_a][ii]-relation_vec[rel_a][ii]);
            if (L1_flag)
                if (x>0)
                    x=1;
                else
                    x=-1;

            relation_vec[rel_a][ii]-=-1*rate*x;
            entity_vec[e1_a][ii]-=-1*rate*x;
            entity_vec[e2_a][ii]+=-1*rate*x;

            x = 2*(entity_vec[e2_b][ii]-entity_vec[e1_b][ii]-relation_vec[rel_b][ii]);

            if (L1_flag)
                if (x>0)
                    x=1;
                else
                    x=-1;

            relation_vec[rel_b][ii]-=rate*x;
            entity_vec[e1_b][ii]-=rate*x;
            entity_vec[e2_b][ii]+=rate*x;
        }
    }
    void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,double &private_res)
    {
        double sum1 = calc_sum(e1_a,e2_a,rel_a);
        double sum2 = calc_sum(e1_b,e2_b,rel_b);

        if (sum1+margin>sum2)
        {
            private_res+=margin+sum1-sum2;
            gradient( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
        }
    }
};

Train train;

void prepare()
{
    FILE* f1 = fopen(("./"+dataset+"/entity2id.txt").c_str(),"r");
    FILE* f2 = fopen(("./"+dataset+"/relation2id.txt").c_str(),"r");

    int x;
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
        left_entity[relation2id[s3]][entity2id[s1]]++;
        right_entity[relation2id[s3]][entity2id[s2]]++;
        train.add(entity2id[s1],entity2id[s2],relation2id[s3]);
    }
    for (int i=0; i<relation_num; i++)
    {
        double sum1=0,sum2=0;
        for (map<int,int>::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
        {
            sum1++;
            sum2+=it->second;
        }
        left_num[i]=sum2/sum1;
    }
    for (int i=0; i<relation_num; i++)
    {
        double sum1=0,sum2=0;
        for (map<int,int>::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
        {
            sum1++;
            sum2+=it->second;
        }
        right_num[i]=sum2/sum1;
    }
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
    fclose(f_kb);
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

        FILE* f = fopen("./all.TransE.log","a");
        fprintf(f,"strategy:%s dataset:%s version:%s nthread:%d k:%d margin:%f nepoch:%d rate:%f \n",strategy.c_str(),dataset.c_str(),version.c_str(),nthread,n,margin,nepoch,rate);
        fprintf(f, "traintime:%d\n", traintime);
        // fprintf(f,"\n");
        fclose(f);

    }
}


