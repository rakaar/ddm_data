function abort_rate_fit(outx)


%% data: abort rate aligned to LED onset

outx(outx.abort_event==3 & outx.TotalFixTime<.3,:) = [];
outx(outx.RTwrtStim<1,:);
outx(outx.RTwrtStim>-2,:);
LED_type = 'LED_bi';
n_aborts_0 = sum(outx.FixAbort==1 & outx.LED==0);
n_valid_0 = sum(outx.success~=0 & outx.LED==0);
n_aborts_1 = sum(outx.FixAbort==1 & outx.(LED_type)==1);
n_valid_1 = sum(outx.success~=0 & outx.(LED_type)==1);

bw = .01;
xx = -1:bw:2;

%%

ff2 = figure('color','w');
col = get(0,'defaultAxesColorOrder');
ff2.Units = 'centimeters';
ff2.Position(1:2) = [3,1.5];
ff2.Position(3:4) = [16,9];
colABL = hex2rgb({'#7D2E8C','#75AB2E','#D95219'});
colLED = hex2rgb('#4DBFED');


ha = tight_subplot(1, 2, [.05 .12], [.17 .05], [.1 .02]);
for k=1:length(ha)
    ha(k).XTickLabelMode = 'auto';
    ha(k).YTickLabelMode = 'auto';
end

%%

axes(ha(1)); hold on;
% why do I need to subtract 0.1? -> I don't
% h_0 = histcounts(outx.TotalFixTime(outx.FixAbort==1 & outx.LED==0)-outx.LED_onset(outx.FixAbort==1 & outx.LED==0),-1:.025:2,'Normalization','pdf');
% h_1 = histcounts(outx.TotalFixTime(outx.FixAbort==1 & outx.LED==1)-outx.LED_onset(outx.FixAbort==1 & outx.LED==1),-1:.025:2,'Normalization','pdf');
h_0 = histcounts(outx.RT_LED_delta(outx.FixAbort==1 & outx.LED==0 & outx.TotalFixTime>=0),xx,'Normalization','pdf');
h_1 = histcounts(outx.RT_LED_delta(outx.FixAbort==1 & outx.(LED_type)==1 & outx.TotalFixTime>=0),xx,'Normalization','pdf');
stairs(xx,[0,h_0]*n_aborts_0/(n_aborts_0+n_valid_0),'color','k')
stairs(xx,[0,h_1]*n_aborts_1/(n_aborts_1+n_valid_1),'color',colLED)
xlabel('abort time aligned to LED onset')
ylabel('abort rate')
ylabel('abort rate (Hz)')
xlim([-1,1])
plot([0,0],[0,1],'k--')





%% model


xmax = .7;
ymax = 1;
pm.dt = 1e-3;
pm.Tmax = 5;            % in sec
pm.Tmin = -3;
tt = pm.Tmin:pm.dt:pm.Tmax;     % row
tt = tt';           % column

% PA params
pm.bound_A = 2.535;
pm.drift_A = 1.55; %1.6
pm.drift_A_ON = 3.4; %3.4

% delay params
pm.delta_E = .075;
pm.delta_A = -.177; %-.155
pm.delta_go = .13;
pm.fix_base = .2;
pm.fix_mean = .4;

delta_m = .045;


    function p = d_A_RT(a,tt)
        % function to calculate the standard PA pdf
        t = tt;
        p = 1./sqrt(2*pi*tt.^3).*exp(-(1-a.*tt).^2./(2*tt));
        p(t<=0) = 0;
    end


    function STF = stupid_f_integral(v,vON,theta,t,tp)
        % function to calculate the PA pdf after the v_A change
        % it calculates the integral of p(x,tp,v,theta)*f(t,vON,a-x)
        % tp and t depend on tstim and tled
        
        dt = t(2)-t(1);
        t_og = t;
        indst = t>0;
        t = t(indst);
        STF = zeros(size(t_og));
        
        a1 = 1/2*(1./t + 1./tp);
        b1 = theta./t + (v-vON);
        %             b1 = -b1;
        c1 = -1/2*(vON^2.*t-2*theta*vON+theta^2./t + v^2*tp);
        
        a2 = a1;
        b2 = theta*(1./t + 2./tp) + (v-vON);
        %             b2 = -b2;
        c2 = -1/2*(vON^2.*t-2*theta*vON+theta^2./t + v^2*tp+4*theta*v+4*theta^2./tp) + 2*v*theta;
        
        F01 = 1./(2*pi*sqrt(tp.*t.^3))./(2*a1);
        F02 = 1./(2*pi*sqrt(tp.*t.^3))./(2*a2);
        
        %b1^2/(4*a1)
        T11 = b1.^2./(4*a1);
        %(2a_1*theta-b1)/(2*sqrt(a1))
        %             T12 = (theta./tp-(v-vON))./(2*sqrt(a1));
        T12 = (2*a1*theta-b1)./(2*sqrt(a1));
        %theta*(b1-theta*a1)
        %             T13 = theta^2/2*(1./t - 1./tp) + theta*(v-vON);
        T13 = theta.*(b1-theta*a1);
        
        %b2^2/(4*a2)
        T21 = b2.^2./(4*a2);
        %(2a_2*theta-b2)/(2*sqrt(a2))
        %             T22 = (-theta./tp-(v-vON))./(2*sqrt(a1));
        T22 = (2*a2*theta-b2)./(2*sqrt(a2));
        %theta*(b2-theta*a2)
        %             T23 = theta^2/2*(1./t + 3./tp) + theta*(v-vON);
        T23 = theta.*(b2-theta*a2);
        
        I1 = F01.*(T12*sqrt(pi).*exp(T11+c1).*(erf(T12)+1) + exp(T13+c1));
        I2 = F02.*(T22*sqrt(pi).*exp(T21+c2).*(erf(T22)+1) + exp(T23+c2));
        
        
        STF(indst) = I1 - I2;
        
    end

   function f_A = PA_with_LEDON_2(t,v,vON,a,tfix,tled)
            
            dt = t(2)-t(1);
            inds1 = t+tfix-pm.delta_m<=tled+dt; 
            inds2 = t+tfix-pm.delta_m>tled+dt;
            t1 = t(inds1);
            t2 = t(inds2);
            
            f_A = zeros(size(t));
            
            f_A(inds1) = d_A_RT(v*a,(t1-pm.delta_A+tfix)/a^2)/a^2;
            
            if ~isempty(t2)
            f_A(inds2) = stupid_f_integral(v,vON,a,t2-pm.delta_m+tfix-tled,t2(1)+pm.delta_m-pm.delta_A+tfix);
            end
            
        end




% simulate fixation and led times (until we have the distribution)
tau = pm.fix_mean;
s = pm.fix_base;



N = 1e4;
tfix = exprnd(tau,N,1)+s;
tled = exprnd(tau/4,N,1)+s/2;
tled = min(tfix,tled);
tled = tfix-tled;

% calculate the integral of f_A with the distribution of tstim and tled (by
% sampling)
tic;
f_A = zeros(size(tt));
f_A2 = zeros(size(tt));
for kt=1:Ns
    f_Ai = PA_with_LEDON_2(tt,pm.drift_A,pm.drift_A_ON,pm.bound_A,tfix(kt),tled(kt),delta_m);
    its = find(tt+tled(kt)>tfix(kt),1);
    f_Ai(its:end) = 0;
    f_A = f_A + f_Ai/N;
    f_Ai2 = PA_with_LEDON_2(tt,pm.drift_A,pm.drift_A,pm.bound_A,tfix(kt),tled(kt),10);
    f_Ai2(its:end) = 0;
    f_A2 = f_A2 + f_Ai2/N;
end

plot(tt,f_A,'color','k')
plot(tt,f_A2,'color',col(1,:))
ylim([0,1])

legend('LED OFF','LED ON')

end


function f_A = PA_with_LEDON_2(t,v,vON,a,tfix,tled,delta)
        % this function just joins the two parts of the PA pdf, before and after LED
        
        dt = t(2)-t(1);
        inds1 = (t-delta)<=dt; %+ o r-?
        inds2 = (t-delta)>dt;
        t1 = t(inds1);
        t2 = t(inds2);
        
        f_A = zeros(size(t));
        
        f_A(inds1) = d_A_RT(v*a,(t1-pm.delta_A+tled)/a^2)/a^2;
        
        if ~isempty(t2)
            f_A(inds2) = stupid_f_integral(v,vON,a,t2-delta,dt-pm.delta_A+delta+tled);
        end
        
    end