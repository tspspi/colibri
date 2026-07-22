/* Exactness test for the integer-dot kernels: dot_i8i8 and dot_i4i8 must return
 * EXACTLY the same value as a plain-C reference, whatever SIMD path was compiled
 * in (avx512-vnni / avx2 / neon / vsx / scalar). Integer arithmetic has no
 * rounding, so any mismatch is a kernel bug, not noise.
 *
 * Covers: odd sizes (scalar tail), sizes below one vector, the w=-128 edge
 * (sign-trick kernels must treat |−128| as 128 unsigned, not saturate to 127),
 * and random data at qrow_i8's contract (|x| <= 127, w full int8 range). */
#define main coli_glm_main_unused
#include "../glm.c"
#undef main

static uint32_t rng_state=0x12345678u;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }

static int32_t ref_i8i8(const int8_t *w, const int8_t *x, int I){
    int64_t s=0; for(int i=0;i<I;i++) s+=(int32_t)w[i]*x[i]; return (int32_t)s;
}
static int32_t ref_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int64_t s=0;
    for(int i=0;i<I;i++){ uint8_t b=w4[i>>1]; int v=(i&1)?((int)(b>>4)-8):((int)(b&0xF)-8); s+=v*x[i]; }
    return (int32_t)s;
}
static float ref_qrow_i8(const float *x, int8_t *q, int I){
    float amax=0;
    for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax) amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}
static int check_qrow(int I){
    float *x=malloc((size_t)I*sizeof(float));
    int8_t *q=malloc((size_t)I), *qref=malloc((size_t)I);
    for(int i=0;i<I;i++) x[i]=((float)((int)(xr()%16385)-8192))/64.f;
    if(I>0) x[0]=0.5f;
    if(I>1) x[1]=-0.5f;
    if(I>2) x[2]=126.5f/127.f;
    if(I>3) x[3]=-126.5f/127.f;
    float s=qrow_i8(x,q,I), sref=ref_qrow_i8(x,qref,I);
    int rc=0;
    if(memcmp(&s,&sref,sizeof(float))!=0 || memcmp(q,qref,(size_t)I)!=0){
        fprintf(stderr,"FAIL qrow_i8 I=%d: scale %.9g != %.9g\n",I,(double)s,(double)sref);
        rc=1;
    }
    free(x); free(q); free(qref);
    return rc;
}

/* Driver-level exactness: matmul_qt_ex on the IDOT path (allow_idot=1) must match
 * a plain-C reference bit-for-bit. This exercises the SMMLA 2x2-tile drivers on the
 * ARCH=native (i8mm) build and the SDOT drivers on the default build. The integer
 * dot is exact and qrow_i8 is the shared quantizer, so any float bit mismatch is a
 * driver bug (lane map, tiling, tail, or scale order), not rounding. */
static void fill_qt(QT *w, int fmt, int O, int I){
    memset(w,0,sizeof *w);
    w->fmt=fmt; w->O=O; w->I=I;
    w->s=malloc((size_t)O*sizeof(float));
    for(int o=0;o<O;o++) w->s[o]=0.001f+(float)(xr()%1000)*1e-6f;
    if(fmt==1){
        w->q8=malloc((size_t)O*I);
        for(int64_t i=0;i<(int64_t)O*I;i++) w->q8[i]=(int8_t)((int)(xr()%256)-128);
    }else{
        size_t nb=(size_t)O*((I+1)/2);
        w->q4=malloc(nb);
        for(size_t i=0;i<nb;i++) w->q4[i]=(uint8_t)(xr()&0xFF);
    }
}
static int check_driver(int fmt,int O,int I,int S){
    QT w; fill_qt(&w,fmt,O,I); int rb=(I+1)/2;
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y=malloc((size_t)S*O*sizeof(float));
    float *yref=malloc((size_t)S*O*sizeof(float));
    int8_t *xqr=malloc((size_t)S*I);
    float *sxr=malloc((size_t)S*sizeof(float));
    for(int64_t i=0;i<(int64_t)S*I;i++) x[i]=((float)(xr()%4001)-2000.f)/500.f;
    matmul_qt_ex(y,x,&w,S,1);
    for(int s=0;s<S;s++) sxr[s]=qrow_i8(x+(int64_t)s*I, xqr+(int64_t)s*I, I);
    for(int o=0;o<O;o++) for(int s=0;s<S;s++){
        int32_t d=fmt==1 ? ref_i8i8(w.q8+(int64_t)o*I, xqr+(int64_t)s*I, I)
                         : ref_i4i8(w.q4+(int64_t)o*rb, xqr+(int64_t)s*I, I);
        yref[(int64_t)s*O+o]=(float)d*w.s[o]*sxr[s];
    }
    int rc=0;
    for(int64_t i=0;i<(int64_t)S*O;i++)
        if(memcmp(&y[i],&yref[i],sizeof(float))!=0){
            fprintf(stderr,"FAIL driver fmt=%d O=%d I=%d S=%d idx=%lld: %.9g != %.9g\n",
                    fmt,O,I,S,(long long)i,(double)y[i],(double)yref[i]); rc=1; break;
        }
    free(w.s); free(w.q8); free(w.q4); free(x); free(y); free(yref); free(xqr); free(sxr);
    return rc;
}

/* Exact attention path: allow_idot=0 must stay on the plain f32-activation int4 kernel and
 * therefore remain bit-identical to the direct dequant reference. This is the real chat-path
 * contract for q_a/q_b/kv_a on CPU. */
static int check_exact_i4_driver(int O,int I,int S){
    QT w; fill_qt(&w,2,O,I);
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y=malloc((size_t)S*O*sizeof(float));
    float *yref=malloc((size_t)S*O*sizeof(float));
    for(int64_t i=0;i<(int64_t)S*I;i++) x[i]=((float)(xr()%4001)-2000.f)/500.f;
    matmul_qt_ex(y,x,&w,S,0);
    matmul_i4(yref,x,w.q4,w.s,S,I,O);
    int rc=0;
    for(int64_t i=0;i<(int64_t)S*O;i++)
        if(memcmp(&y[i],&yref[i],sizeof(float))!=0){
            fprintf(stderr,"FAIL exact i4 O=%d I=%d S=%d idx=%lld: %.9g != %.9g\n",
                    O,I,S,(long long)i,(double)y[i],(double)yref[i]);
            rc=1;
            break;
        }
    free(w.s); free(w.q4); free(x); free(y); free(yref);
    return rc;
}

static void ref_rope_interleave(float *v, int pos, const Cfg *c){
    int half=c->qk_rope/2;
    float in[256];
    memcpy(in,v,(size_t)c->qk_rope*sizeof(float));
    for(int j=0;j<half;j++){
        float inv=powf(c->theta,-2.0f*j/c->qk_rope), ang=pos*inv;
        float cs=cosf(ang), sn=sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        v[j]      = a*cs - b*sn;
        v[half+j] = b*cs + a*sn;
    }
}

static int check_rope_exact(int qk, int pos, float theta){
    Cfg c={0};
    c.qk_rope=qk;
    c.theta=theta;
    float a[256], b[256];
    for(int i=0;i<qk;i++) a[i]=b[i]=((float)(xr()%4001)-2000.f)/500.f;
    g_no_rope_inv_cache=0;
    rope_interleave(a,pos,&c);
    ref_rope_interleave(b,pos,&c);
    if(memcmp(a,b,(size_t)qk*sizeof(float))!=0){
        for(int i=0;i<qk;i++) if(memcmp(&a[i],&b[i],sizeof(float))!=0){
            fprintf(stderr,"FAIL rope qk=%d pos=%d theta=%.9g idx=%d: %.9g != %.9g\n",
                    qk,pos,(double)theta,i,(double)a[i],(double)b[i]);
            break;
        }
        return 1;
    }
    return 0;
}

static void ref_qt_addrow(const QT *t, int row, float coef, float *acc){
    int I=t->I;
    if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=coef*w[i]; return; }
    if(t->fmt==4){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
        int gs=t->gs, ng=(I+gs-1)/gs; const float *scl=t->s+(int64_t)row*ng;
        for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1];
            acc[i]  +=coef*scl[i/gs]    *((int)(b&0xF)-8);
            acc[i+1]+=coef*scl[(i+1)/gs]*((int)(b>>4)-8); }
        if(I&1){ uint8_t b=w[I>>1]; acc[I-1]+=coef*scl[(I-1)/gs]*((int)(b&0xF)-8); }
        return;
    }
    float c=coef*t->s[row];
    if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=c*(float)w[i]; return; }
    if(t->fmt==2){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
        for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc[i]+=c*((int)(b&0xF)-8); acc[i+1]+=c*((int)(b>>4)-8); }
        if(I&1){ uint8_t b=w[I>>1]; acc[I-1]+=c*((int)(b&0xF)-8); }
        return;
    }
    { const uint8_t *w=t->q4+(int64_t)row*((I+3)/4);
        for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; acc[i]+=c*((int)((b>>((i&3)*2))&3)-2); } }
}

static int check_qt_addrow_exact(int fmt,int O,int I,int row,float coef){
    QT w={0};
    w.fmt=fmt; w.O=O; w.I=I;
    if(fmt==0){
        w.qf=malloc((size_t)O*I*sizeof(float));
        for(int64_t i=0;i<(int64_t)O*I;i++) w.qf[i]=((float)(xr()%4001)-2000.f)/500.f;
    } else if(fmt==1){
        w.q8=malloc((size_t)O*I);
        w.s=malloc((size_t)O*sizeof(float));
        for(int o=0;o<O;o++) w.s[o]=0.001f+(float)(xr()%1000)*1e-6f;
        for(int64_t i=0;i<(int64_t)O*I;i++) w.q8[i]=(int8_t)((int)(xr()%256)-128);
    } else if(fmt==2){
        w.q4=malloc((size_t)O*((I+1)/2));
        w.s=malloc((size_t)O*sizeof(float));
        for(int o=0;o<O;o++) w.s[o]=0.001f+(float)(xr()%1000)*1e-6f;
        for(int64_t i=0;i<(int64_t)O*((I+1)/2);i++) w.q4[i]=(uint8_t)(xr()&0xFF);
    } else {
        int gs=64, ng=(I+gs-1)/gs;
        w.gs=gs;
        w.q4=malloc((size_t)O*((I+1)/2));
        w.s=malloc((size_t)O*ng*sizeof(float));
        for(int64_t i=0;i<(int64_t)O*((I+1)/2);i++) w.q4[i]=(uint8_t)(xr()&0xFF);
        for(int64_t i=0;i<(int64_t)O*ng;i++) w.s[i]=0.001f+(float)(xr()%1000)*1e-6f;
    }
    float *a=malloc((size_t)I*sizeof(float));
    float *b=malloc((size_t)I*sizeof(float));
    for(int i=0;i<I;i++) a[i]=b[i]=((float)(xr()%4001)-2000.f)/500.f;
    qt_addrow(&w,row,coef,a);
    ref_qt_addrow(&w,row,coef,b);
    int rc=0;
    for(int i=0;i<I;i++) if(memcmp(&a[i],&b[i],sizeof(float))!=0){
        fprintf(stderr,"FAIL qt_addrow fmt=%d O=%d I=%d row=%d coef=%.9g idx=%d: %.9g != %.9g\n",
                fmt,O,I,row,(double)coef,i,(double)a[i],(double)b[i]);
        rc=1;
        break;
    }
    free(w.qf); free(w.q8); free(w.q4); free(w.s); free(a); free(b);
    return rc;
}

static void ref_rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps);
    for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}

static int check_rmsnorm_exact(int D, float eps){
    float *x=malloc((size_t)D*sizeof(float));
    float *w=malloc((size_t)D*sizeof(float));
    float *a=malloc((size_t)D*sizeof(float));
    float *b=malloc((size_t)D*sizeof(float));
    for(int i=0;i<D;i++){
        x[i]=((float)(xr()%4001)-2000.f)/500.f;
        w[i]=((float)(xr()%4001)-2000.f)/500.f;
    }
    g_no_rmsnorm_simd=0;
    rmsnorm(a,x,w,D,eps);
    ref_rmsnorm(b,x,w,D,eps);
    for(int i=0;i<D;i++) if(memcmp(&a[i],&b[i],sizeof(float))!=0){
        fprintf(stderr,"FAIL rmsnorm D=%d eps=%.9g idx=%d: %.9g != %.9g\n",
                D,(double)eps,i,(double)a[i],(double)b[i]);
        free(x); free(w); free(a); free(b);
        return 1;
    }
    free(x); free(w); free(a); free(b);
    return 0;
}

static int check_expert_gate_up_small_i8_exact(int O,int I,int S){
    QT wg={0}, wu={0};
    fill_qt(&wg,1,O,I);
    fill_qt(&wu,1,O,I);
    float *x=malloc((size_t)S*I*sizeof(float));
    float *g0=malloc((size_t)S*O*sizeof(float));
    float *u0=malloc((size_t)S*O*sizeof(float));
    float *g1=malloc((size_t)S*O*sizeof(float));
    float *u1=malloc((size_t)S*O*sizeof(float));
    for(int64_t i=0;i<(int64_t)S*I;i++) x[i]=((float)(xr()%4001)-2000.f)/500.f;
    g_no_i8_small_reuse=0;
    expert_gate_up(g0,u0,x,&wg,&wu,S);
    g_no_i8_small_reuse=1;
    expert_gate_up(g1,u1,x,&wg,&wu,S);
    for(int64_t i=0;i<(int64_t)S*O;i++) if(memcmp(&g0[i],&g1[i],sizeof(float))!=0){
        fprintf(stderr,"FAIL expert_gate_up small-i8 gate O=%d I=%d S=%d idx=%lld: %.9g != %.9g\n",
                O,I,S,(long long)i,(double)g0[i],(double)g1[i]);
        free(wg.s); free(wg.q8); free(wu.s); free(wu.q8); free(x); free(g0); free(u0); free(g1); free(u1);
        return 1;
    }
    for(int64_t i=0;i<(int64_t)S*O;i++) if(memcmp(&u0[i],&u1[i],sizeof(float))!=0){
        fprintf(stderr,"FAIL expert_gate_up small-i8 up O=%d I=%d S=%d idx=%lld: %.9g != %.9g\n",
                O,I,S,(long long)i,(double)u0[i],(double)u1[i]);
        free(wg.s); free(wg.q8); free(wu.s); free(wu.q8); free(x); free(g0); free(u0); free(g1); free(u1);
        return 1;
    }
    free(wg.s); free(wg.q8); free(wu.s); free(wu.q8); free(x); free(g0); free(u0); free(g1); free(u1);
    return 0;
}

int main(void){
    static const int sizes[]={1,2,15,16,17,31,32,33,63,64,65,100,127,128,1408,4096,4097};
    static int8_t w[8192], x[8192]; static uint8_t w4[4096];
    for(unsigned t=0;t<sizeof(sizes)/sizeof(sizes[0]);t++)
        for(int rep=0;rep<32;rep++)
            if(check_qrow(sizes[t])) return 1;
    printf("qrow_i8 exactness: ok\n");
    for(unsigned t=0;t<sizeof(sizes)/sizeof(sizes[0]);t++){
        int I=sizes[t];
        for(int rep=0;rep<64;rep++){
            for(int i=0;i<I;i++) x[i]=(int8_t)((int)(xr()%255)-127);      /* [-127,127]: contratto di qrow_i8 */
            for(int i=0;i<I;i++) w[i]=(int8_t)((int)(xr()%256)-128);      /* [-128,127]: range pieno */
            if(rep==0) for(int i=0;i<I;i++) w[i]=-128;                    /* caso limite del trucco del segno */
            if(rep==1) for(int i=0;i<I;i++){ w[i]=127; x[i]=(int8_t)(i&1?-127:127); }
            for(int i=0;i<(I+1)/2;i++) w4[i]=(uint8_t)(xr()&0xFF);
            int32_t got=dot_i8i8(w,x,I), want=ref_i8i8(w,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i8i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
            got=dot_i4i8(w4,x,I); want=ref_i4i8(w4,x,I);
            if(got!=want){ fprintf(stderr,"FAIL dot_i4i8 I=%d rep=%d: %d != %d\n",I,rep,got,want); return 1; }
        }
    }
    printf("idot kernel exactness (%s): ok\n", IDOT_KERNEL);

    static const int Os[]={1,2,3,64,65};
    static const int Is[]={16,17,100,1408};
    static const int Ss[]={2,3,4,5,8};
    for(int rep=0;rep<4;rep++)
     for(unsigned a=0;a<sizeof Os/sizeof Os[0];a++)
      for(unsigned b=0;b<sizeof Is/sizeof Is[0];b++)
       for(unsigned c=0;c<sizeof Ss/sizeof Ss[0];c++)
        for(int fmt=1;fmt<=2;fmt++)
         if(check_driver(fmt,Os[a],Is[b],Ss[c])) return 1;
    printf("idot driver exactness (%s): ok\n", IDOT_KERNEL);

    static const int exact_Os[]={1,2,3,64};
    static const int exact_Is[]={16,17,100,1408};
    static const int exact_Ss[]={1,2,5,36};
    for(int rep=0;rep<4;rep++)
     for(unsigned a=0;a<sizeof exact_Os/sizeof exact_Os[0];a++)
      for(unsigned b=0;b<sizeof exact_Is/sizeof exact_Is[0];b++)
       for(unsigned c=0;c<sizeof exact_Ss/sizeof exact_Ss[0];c++)
        if(check_exact_i4_driver(exact_Os[a],exact_Is[b],exact_Ss[c])) return 1;
    printf("exact i4 attention path: ok\n");

    static const int rope_qks[]={2,16,64,128};
    static const int rope_poss[]={0,1,7,35,4095};
    static const float rope_thetas[]={10000.f,50000.f,500000.f};
    for(int rep=0;rep<4;rep++)
     for(unsigned a=0;a<sizeof rope_qks/sizeof rope_qks[0];a++)
      for(unsigned b=0;b<sizeof rope_poss/sizeof rope_poss[0];b++)
       for(unsigned c=0;c<sizeof rope_thetas/sizeof rope_thetas[0];c++)
        if(check_rope_exact(rope_qks[a],rope_poss[b],rope_thetas[c])) return 1;
    printf("rope exactness: ok\n");

    static const int addrow_fmts[]={0,1,2,4};
    static const int addrow_Is[]={15,16,17,64,65,192,512};
    static const float addrow_cs[]={-2.5f,-0.125f,0.0f,0.75f,3.0f};
    for(int rep=0;rep<4;rep++)
     for(unsigned a=0;a<sizeof addrow_fmts/sizeof addrow_fmts[0];a++)
      for(unsigned b=0;b<sizeof addrow_Is/sizeof addrow_Is[0];b++)
       for(unsigned c=0;c<sizeof addrow_cs/sizeof addrow_cs[0];c++)
        if(check_qt_addrow_exact(addrow_fmts[a],3,addrow_Is[b],rep%3,addrow_cs[c])) return 1;
    printf("qt_addrow exactness: ok\n");

    static const int rms_Ds[]={16,17,64,65,512,2048,6144};
    static const float rms_epss[]={1e-6f,1e-5f,1e-4f};
    for(int rep=0;rep<4;rep++)
     for(unsigned a=0;a<sizeof rms_Ds/sizeof rms_Ds[0];a++)
      for(unsigned b=0;b<sizeof rms_epss/sizeof rms_epss[0];b++)
       if(check_rmsnorm_exact(rms_Ds[a],rms_epss[b])) return 1;
    printf("rmsnorm exactness: ok\n");

    static const int eg_Os[]={1,2,3,64};
    static const int eg_Is[]={16,17,100,1408};
    static const int eg_Ss[]={1,2,3};
    for(int rep=0;rep<4;rep++)
     for(unsigned a=0;a<sizeof eg_Os/sizeof eg_Os[0];a++)
      for(unsigned b=0;b<sizeof eg_Is/sizeof eg_Is[0];b++)
       for(unsigned c=0;c<sizeof eg_Ss/sizeof eg_Ss[0];c++)
        if(check_expert_gate_up_small_i8_exact(eg_Os[a],eg_Is[b],eg_Ss[c])) return 1;
    printf("expert gate/up small-i8 exactness: ok\n");

    return 0;
}
