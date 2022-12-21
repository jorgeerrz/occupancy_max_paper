### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 63a9f3aa-31a8-11ec-3238-4f818ccf6b6c
begin
	#using Pkg
	#Pkg.activate("../Project.toml")
	using Plots, Distributions, PlutoUI, Parameters
	using Interpolations, StatsPlots, DelimitedFiles, ZipFile
end

# ╔═╡ 1525f483-b6ac-4b12-90e6-31e97637d282
using StatsBase

# ╔═╡ 18382e9c-e812-49ff-84cc-faad709bc4c3
using LinearAlgebra

# ╔═╡ 11f37e01-7958-4e79-a8bb-06b92d7cb9ed
begin
	PlutoUI.TableOfContents(aside = true)
end

# ╔═╡ 0a29d50e-df0a-4f54-b588-faa6aee2e983
theme(:default,titlefont = ("Computer Modern",16), legend_font_family = "Computer Modern", legend_font_pointsize = 14, guidefont = ("Computer Modern", 16), tickfont = ("Computer Modern", 16),foreground_color_border = :black)

# ╔═╡ 379318a2-ea2a-4ac1-9046-0fdfe8c102d4
interpolation = true

# ╔═╡ 40ee8d98-cc45-401e-bd9d-f9002bc4beba
md"# Useful functions"

# ╔═╡ 4ec0dd6d-9d3d-4d30-8a19-bc91600d9ec2
@with_kw mutable struct State
	θ::Float64 = 0
	w::Float64 = 0
	u::Float64 = 1
	v::Float64 = 0
	x::Float64 = 0
end

# ╔═╡ f997fa16-69e9-4df8-bed9-7067e1a5537d
function dwdt(θ,w,force,env)
	num = env.g*sin(θ) + env.α*cos(θ)*force - env.m*env.α*w^2*env.l*sin(2*θ)/2
	den = env.l*(4/3 -env.m*env.α*cos(θ)^2)
	num/den
end

# ╔═╡ 2f31b286-11aa-443e-a3dc-c021e6fc276c
function searchsortednearest(a,x,which)
	idx = searchsortedfirst(a,x)
	if (idx==1); return idx; end
	if (idx>length(a)); return length(a); end
	if (a[idx]==x); return idx; end
	if which == "position"
		#For ceiling interpolation
		if sign(x) > 0 
		#For nearest neighbour interpolation
		#if (abs(a[idx]-x) < abs(a[idx-1]-x))
	      return idx
	   else
	      return idx-1
	   end
	elseif which == "speed"
		#For ceiling interpolation
		if sign(x) < 0 
		#For nearest neighbour interpolation
		#if (abs(a[idx]-x) < abs(a[idx-1]-x))
	      return idx
	   else
	      return idx-1
	   end
	else
		#For ceiling interpolation
		#if sign(x) > 0 
		#For nearest neighbour interpolation
		if (abs(a[idx]-x) < abs(a[idx-1]-x))
	      return idx
	   else
	      return idx-1
	   end
	end
end

# ╔═╡ 9396a0d1-6036-44ec-98b7-16df4d150b54
md"# H agent"

# ╔═╡ cfdf3a8e-a377-43ef-8a76-63cf78ce6a16
@with_kw struct inverted_pendulum_borders
	g::Float64 = 9.81
	l::Float64 = 1
	M::Float64 = 1
	m::Float64 = 0.1
	γ::Float64 = 0.99
	α::Float64 = 1/(M+m)
	Δt::Float64 = 0.01
	sizeu :: Int64 = 2
	sizev :: Int64 = 21
	sizew :: Int64 = 21
	sizex :: Int64 = 21
	sizeθ :: Int64 = 51
	nstates :: Int64 = sizeθ*sizew*sizeu*sizev*sizex
	nactions :: Int64 = 5
	max_a = 20
	a_s = collect(-max_a:2*max_a/(nactions-1):max_a)
	max_θ = 0.78 #pi/4
	max_w :: Float64 = 3
	max_v :: Float64 = 1/0.1
	max_x :: Float64 = 5
	#sq = uniform.^2 .*sign.(uniform)
	#sq ./= abs(uniform[1])
	θs = collect(-max_θ:2*max_θ/(sizeθ-1):max_θ)
	ws = collect(-max_w:2*max_w/(sizew-1):max_w)
	vs = collect(-max_v:2*max_v/(sizev-1):max_v)
	xs = collect(-max_x:2*max_x/(sizex-1):max_x)
	#Smaller values get more precision
	# θs = (collect(-max_θ:2*max_θ/(sizeθ-1):max_θ).^2 .* sign.(collect(-max_θ:2*max_θ/(sizeθ-1):max_θ)))/max_θ
	# ws = (collect(-max_w:2*max_w/(sizew-1):max_w).^2 .* sign.(collect(-max_w:2*max_w/(sizew-1):max_w)))/max_w
	# vs = (collect(-max_v:2*max_v/(sizev-1):max_v).^2 .* sign.(collect(-max_v:2*max_v/(sizev-1):max_v)))/max_v
	# xs = (collect(-max_x:2*max_x/(sizex-1):max_x).^2 .* sign.(collect(-max_x:2*max_x/(sizex-1):max_x)))/max_x
end

# ╔═╡ b1d3fff1-d980-4cc8-99c3-d3db7a71bf60
function real_transition(state::State,action,env::inverted_pendulum_borders)
	acc = dwdt(state.θ,state.w,action,env)
	new_th = state.θ + state.w*env.Δt #+ acc*env.Δt^2/2
	new_w = state.w + acc*env.Δt
	#According to the paper, but there is a sign error
	#acc_x = env.α*(action + env.m*env.l*(state.w^2*sin(state.θ)-acc*cos(state.θ)))
	acc_x = (4/3 * env.l * acc - env.g*sin(state.θ))/cos(state.θ)
	new_v = state.v + acc_x*env.Δt
	new_x = state.x + state.v*env.Δt #+ acc_x*env.Δt^2/2
	new_u = state.u
	if abs(new_th) >= env.max_θ 
		new_th = sign(new_th)*env.max_θ
		new_u = 1
	end
	if abs(new_x) >= env.max_x
		new_x = sign(new_x)*env.max_x
		new_u = 1
	end
	if abs(new_v) >=env.max_v
		new_v = sign(new_v)*env.max_v
	end
	if abs(new_w) >= env.max_w
		new_w = sign(new_w)*env.max_w
	end
	State(θ = new_th, w = new_w, v = new_v, x = new_x, u = new_u)
end

# ╔═╡ ffe32878-8732-46d5-b57c-9e9bb8e6dd74
function adm_actions_b(s::State, ip::inverted_pendulum_borders)
	out = ip.a_s
	ids = collect(1:ip.nactions)
	#If dead, only no acceleration
	if s.u < 2 
		out = [0]
		ids = [Int((ip.nactions+1)/2)]
	end
	out,Int.(ids)
end

# ╔═╡ 784178c3-4afc-4c65-93e1-4265e1faf817
function build_nonflat_index_b(state::State,env::inverted_pendulum_borders)
	idx_th = searchsortednearest(env.θs,state.θ,"position")
	idx_w = searchsortednearest(env.ws,state.w,"normal")
	idx_u = state.u
	idx_v = searchsortednearest(env.vs,state.v,"normal")
	idx_x = searchsortednearest(env.xs,state.x,"position")
	[idx_th,idx_w,idx_u,idx_v,idx_x]
end

# ╔═╡ fa56d939-09d5-4b63-965a-38016c957fbb
function build_index_b(state_index,env::inverted_pendulum_borders)
	Int64(state_index[1] + env.sizeθ*(state_index[2]-1) + env.sizeθ*env.sizew*(state_index[3]-1)+ env.sizeθ*env.sizew*env.sizeu*(state_index[4]-1)+ env.sizeθ*env.sizew*env.sizeu*env.sizev*(state_index[5]-1))
end

# ╔═╡ 05b8a93c-cefe-473a-9f8e-3e82de0861b2
function transition_b(state::State,action,env::inverted_pendulum_borders)
	if state.u > 1
		acc = dwdt(state.θ,state.w,action,env)
		new_th = state.θ + state.w*env.Δt #+ acc*env.Δt^2/2
		new_w = state.w + acc*env.Δt
		#acc_x = env.α*(action + env.m*env.l*(state.w^2*sin(state.θ)-acc*cos(state.θ)))
		acc_x = (4/3 *env.l * acc - env.g*sin(state.θ))/cos(state.θ)
		new_v = state.v + acc_x*env.Δt
		new_x = state.x + state.v*env.Δt #+ acc_x*env.Δt^2/2
		idx_th = searchsortednearest(env.θs,new_th,"position")
		idx_w = searchsortednearest(env.ws,new_w,"normal")
		idx_v = searchsortednearest(env.vs,new_v,"normal")
		idx_x = searchsortednearest(env.xs,new_x,"position")
		new_u= state.u
		#If angle or location pass threshold, cartpole dies
		if abs(env.θs[idx_th]) >= env.max_θ || abs(env.xs[idx_x]) >= env.max_x
		#if abs(new_th) >= env.max_θ || abs(new_x) >= env.max_x
			new_u = 1
		end
	else
		new_th = state.θ
		new_w = state.w
		new_v = state.v
		new_x = state.x
		new_u = state.u
		idx_th = searchsortednearest(env.θs,state.θ,"position")
		idx_w = searchsortednearest(env.ws,state.w,"normal")
		idx_v = searchsortednearest(env.vs,state.v,"normal")
		idx_x = searchsortednearest(env.xs,state.x,"position")		
	end
	θ_new = env.θs[idx_th]
	w_new = env.ws[idx_w]
	v_new = env.vs[idx_v]
	x_new = env.xs[idx_x]
	[idx_th,idx_w,new_u,idx_v,idx_x],State(θ = θ_new, w = w_new, v = v_new, u = new_u, x = x_new)
end

# ╔═╡ f92e8c6d-bf65-4786-8782-f38847a7fb7a
function discretize_state(state::State,env::inverted_pendulum_borders)
	idx_th = searchsortednearest(env.θs,state.θ,"position")
	idx_w = searchsortednearest(env.ws,state.w,"normal")
	idx_v = searchsortednearest(env.vs,state.v,"normal")
	idx_x = searchsortednearest(env.xs,state.x,"position")
	[idx_th,idx_w,state.u,idx_v,idx_x],State(θ = env.θs[idx_th], w = env.ws[idx_w], u = state.u, v = env.vs[idx_v], x = env.xs[idx_x])
end

# ╔═╡ f9121267-8d7e-40b1-b9cc-c8da3f45cdb8
function reachable_states_b(state::State,action,env::inverted_pendulum_borders)
	[transition_b(state,action,env)[1]],[transition_b(state,action,env)[2]]
end

# ╔═╡ 36601cad-1ba9-48f2-8463-a58f98bedd34
function degeneracy_cost(state,env::inverted_pendulum_borders,δ = 1E-5)
	-δ*((state.θ/env.max_θ)^2+(state.x/env.max_x)^2+(state.v/env.max_v)^2+(state.w/env.max_w)^2)/4
end

# ╔═╡ b975eaf8-6a94-4e39-983f-b0fb58dd70a1
function optimal_policy_b(state,value,env::inverted_pendulum_borders)
	#Check admissible actions
	actions,ids_actions = adm_actions_b(state,env)
	policy = zeros(length(actions))
	#state_index = build_nonflat_index_b(state,env)
	#id_state = build_index_b(state_index,env)
	#Value at current state
	v_state = value(state.θ,state.w,state.u,state.v,state.x)
	for (idx,a) in enumerate(actions)
		#Given action, check reachable states (deterministic environment for now)
		#s_primes_ids,states_p = reachable_states_b(state,a,env)
		s_prime = real_transition(state,a,env)
		exponent = 0
		#for s_p in s_primes_ids
			P = 1
			#if interpolation == true
				#interpolated value
				exponent += env.γ*P*value(s_prime.θ,s_prime.w,s_prime.u,s_prime.v,s_prime.x)
			# else
			# 	#deterministic environment
			# 	i_p = build_index_b(s_primes_ids[1],env)
			# 	exponent += env.γ*P*value[i_p]
			# end
		#end
		#Normalize at exponent
		exponent -= v_state
		policy[idx] = exp(exponent)
	end
	#Since we are using interpolated values, policy might not be normalized, so we normalize
	#println("sum policy = ", sum(policy))
	policy = policy./sum(policy)
	#Return available actions and policy distribution over actions
	actions,policy
end

# ╔═╡ e0fdce26-41b9-448a-a86a-6b29b68a6782
function interpolate_value(flat_value,env::inverted_pendulum_borders)
	value_reshaped = reshape(flat_value,length(env.θs),length(env.ws),2,length(env.vs),length(env.xs))
	θs = -env.max_θ:2*env.max_θ/(env.sizeθ-1):env.max_θ
	ws = -env.max_w:2*env.max_w/(env.sizew-1):env.max_w
	vs = -env.max_v:2*env.max_v/(env.sizev-1):env.max_v
	xs = -env.max_x:2*env.max_x/(env.sizex-1):env.max_x
	itp = Interpolations.interpolate(value_reshaped,BSpline(Linear()))
	sitp = Interpolations.scale(itp,θs,ws,1:2,vs,xs)
	sitp
end

# ╔═╡ ce8bf897-f240-44d8-ae39-cf24eb115704
function iteration(env::inverted_pendulum_borders, tolerance = 1E0, n_iter = 100;δ = 1E-5,verbose = false)
	v = zeros(env.nstates)
	v_new = zeros(env.nstates)
	t_stop = n_iter
	error = 0
	f_error = 0
	for t in 1:n_iter
		v_itp = interpolate_value(v,env)
		ferror_max = 0
		#Parallelization over states
		Threads.@threads for idx_θ in 1:env.sizeθ
			for idx_w in 1:env.sizew, idx_u in 1:env.sizeu, idx_v in 1:env.sizev, idx_x in 1:env.sizex
				state_idx = [idx_θ,idx_w,idx_u,idx_v,idx_x]
				state = State(θ = env.θs[idx_θ], w = env.ws[idx_w], u = idx_u, v = env.vs[idx_v], x = env.xs[idx_x])
				i = build_index_b(state_idx,env)
				actions,ids_actions = adm_actions_b(state,env)
				sum = 0
				# Add negligible external reward that breaks degeneracy for 
				# Q agent
				small_reward = degeneracy_cost(state,env,δ)
				for (id_a,a) in enumerate(actions)
					#For every action, look at reachable states
					#s_primes_ids,states_p = reachable_states_b(state,a,env)
					states_p = [real_transition(state,a,env)]
					exponent = 0
					for (idx,s_p) in enumerate(states_p)
						#i_p = build_index_b(s_p,env)
						P = 1
						exponent += env.γ*P*v_itp(s_p.θ,s_p.w,s_p.u,s_p.v,s_p.x)
					end
					sum += exp(exponent + small_reward)
				end
				v_new[i] = log(sum)
				f_error = abs(v[i] - v_new[i])
				# Use supremum norm of difference between values
				# at different iterations
				ferror_max = max(ferror_max,f_error)
			end
		end
		# Check overall error between value's values at different iterations
		error = norm(v-v_new)/norm(v)
		#if f_error < tolerance
		if ferror_max < tolerance
			t_stop = t
			break
		end
		if verbose == true
		println("iteration = ", t, ", error = ", error, ", max function error = ", ferror_max)
		end
		v = deepcopy(v_new)
	end
	v_new,error,t_stop
end

# ╔═╡ bbe19356-e00b-4d90-b704-db33e0b75743
ip_b = inverted_pendulum_borders(M = 1, m = 0.1,l = 1.,Δt = 0.02, sizeθ = 31, sizew = 31,sizev = 31, sizex = 31, max_θ = 0.62, max_w = 3, a_s = [-40,-10,0,10,40], max_x = 1.8, max_v = 6, nactions = 5, γ = 0.98)

# ╔═╡ b38cdb2b-f570-41f4-8996-c7e41551f374
function draw_cartpole(xposcar,xs,ys,t,xlimit,ip::inverted_pendulum_borders)
	hdim = 400
	vdim = 200
	size = 2
	verts = [(-size,-size/2),(size,-size/2),(size,size/2),(-size,size/2)]
	#Draw car
	pcar = plot(xticks = false, yticks = false,xlim = (-xlimit-0.3,xlimit+0.3),ylim = (-0.2, ip.l + 0.05), grid = false, axis = false,legend = false)
	#Plot arena
	plot!(pcar, [-xlimit-0.2,xlimit+0.2], [-0.1,-0.1], color = :black)
	plot!(pcar, [-xlimit-0.3,-xlimit-0.2], [0.2*ip.l,0.2*ip.l], color = :black)
	plot!(pcar, [xlimit+0.2,xlimit+0.3], [0.2*ip.l,0.2*ip.l], color = :black)
	plot!(pcar, [-xlimit-0.2,-xlimit-0.2], [-0.1,0.2*ip.l], color = :black)
	plot!(pcar, [xlimit+0.2,xlimit+0.2], [-0.1,0.2*ip.l], color = :black)
	#Draw pole
	plot!(pcar,xs[t],ys[t],marker = (:none, 1),linewidth = 2., linecolor = :black)
	#Draw cart
	plot!(pcar,xposcar[t],[0.04],marker = (Shape(verts),26),mswidth = 0,color = :pink)
	#draw_actions
	actions,_ = adm_actions_b(State(u = 2),ip)
	arrow_x = zeros(length(actions))
	arrow_y = zeros(length(actions))
	aux = actions
	for i in 1:length(aux)
		mult = 0.03
		if i == 1 || i == 5
			mult = 0.02
		end
		arrow_x[i] = aux[i]*mult
		arrow_y[i] = 0#aux[i][2]*mult*1.3
	end
	quiver!(pcar,ones(Int64,length(aux))*xposcar[t][1],[0.01,0.06,0.04,0.06,0.01],quiver = (arrow_x,arrow_y),color = "black",linewidth = 2)
	scatter!(pcar,xposcar[t],[0.04],markersize = 10,color = "black")
	#draw middle of arena
	scatter!(pcar,[0.0],[-0.14],markershape = :vline,color = "black",markersize = 5)
	plot(pcar, size = (hdim,vdim),margin=4Plots.mm)
	
	
end

# ╔═╡ 79372251-157d-43a0-9560-4727fbd36ea9
function draw_cartpole_time(xposcar,xs,ys,xlimit,maxtime,step,ip::inverted_pendulum_borders)
	hdim = 500
	vdim = 200
	size = 1
	verts = [(-size,-size/2),(size,-size/2),(size,size/2),(-size,size/2)]
	#Draw car
	pcar = plot(xticks = false, yticks = false,xlim = (-xlimit-0.3,xlimit+0.3),ylim = (-0.1, ip.l + 0.05), grid = false, axis = false,legend = false)
	#Plot arena
	plot!(pcar, [-xlimit-0.2,xlimit+0.2], [-0.1,-0.1], color = :black)
	plot!(pcar, [-xlimit-0.3,-xlimit-0.2], [0.2*ip.l,0.2*ip.l], color = :black)
	plot!(pcar, [xlimit+0.2,xlimit+0.3], [0.2*ip.l,0.2*ip.l], color = :black)
	plot!(pcar, [-xlimit-0.2,-xlimit-0.2], [-0.1,0.2*ip.l], color = :black)
	plot!(pcar, [xlimit+0.2,xlimit+0.2], [-0.1,0.2*ip.l], color = :black)
	for t in 1:step:maxtime
		#Draw pole
		plot!(pcar,xs[t],ys[t],marker = (:none, 1),linewidth = 1, linecolor = :black,alpha = 0.1)
		#Draw cart
		plot!(pcar,xposcar[t],[0],marker = (Shape(verts),30),alpha = 0.1 ,color = :pink)
	end
	plot(pcar, size = (hdim,vdim),margin=6Plots.mm)
end

# ╔═╡ 10a47fa8-f235-4455-892f-9b1457b1f82c
function draw_cartpole_timeright(xposcar,xs,ys,maxtime,step,ip::inverted_pendulum_borders)
	hdim = 500
	vdim = 100
	size_cart = 0.1
	Δx = 1/maxtime #0.01#hdim/maxtime#0.007/step
	verts = [(-size_cart,-size_cart/2),(size_cart,-size_cart/2),(size_cart,size_cart/2),(-size_cart,size_cart/2)]
	#Initialize car plot
	pcar = plot(xticks = false, yticks = false,xlim = (0,maxtime*Δx),ylim = (-0.1, ip.l + 0.05), grid = false, axis = false,legend = false)
	for t in 1:step:maxtime
		#Draw pole
		#plot!(pcar,[0,xs[t][2]-xs[t][1]]+[t*Δx,t*Δx],ys[t],marker = (:none, 1),linewidth = 1.2, linecolor = :black,alpha = 0.3)
		plot!(pcar,[0,xs[t][2]-xs[t][1]]*0.2+[t*Δx,t*Δx],ys[t],marker = (:none, 1),linewidth = 1.2, linecolor = :black,alpha = 0.3)
		#Draw cart
		plot!(pcar,0*xposcar[t] +[t*Δx],[0],marker = (Shape(verts),30),alpha = 0.5 ,color = "#A05A2C")
	end
	plot(pcar, size = (hdim,vdim),margin=1Plots.mm)
end

# ╔═╡ e099528b-37a4-41a2-836b-69cb3ceda2f5
md" ## Value iteration"

# ╔═╡ 6b9b1c38-0ed2-4fe3-9326-4670a33e7765
#Tolerance for iteration, supremum norm
tolb = 1E-1

# ╔═╡ 7c04e2c3-4157-446d-a065-4bfe7d1931fd
begin
#To calculate value, uncomment, it takes around 30 minutes for 1.8E6 states
#h_value,error,t_stop= iteration(ip_b,tolb,1200,δ = δ_r);
#writedlm("h_value_g_$(ip_q.γ)_nstates_$(ip_b.nstates).dat",h_value)
#Read from compressed file
	# h_zip = ZipFile.Reader("h_value_g_$(ip_q.γ)_nstates_$(ip_q.nstates).dat.zip")
	# h_value = readdlm(h_zip.files[1], Float64)
	#h_value = readdlm("values_borders/h_value_nstates_$(ip_b.nstates)_xlim_2.4.dat");
	h_value = readdlm("values/h_value_g_$(ip_b.γ)_nstates_$(ip_b.nstates).dat");
end;

# ╔═╡ a0b85a14-67af-42d6-b231-1c7d0c293f6e
#If value calculated, this code stores the value in a dat file
#writedlm("values/h_value_g_$(ip_b.γ)_nstates_$(ip_b.nstates).dat",h_value)

# ╔═╡ 8a59e209-9bb6-4066-b6ca-70dac7da33c3
h_value_int = interpolate_value(h_value,ip_b);

# ╔═╡ 08ecfbed-1a7c-43d4-ade7-bf644eeb6eee
function create_episode_b(state_0,int_value,max_t,env::inverted_pendulum_borders)
	x = 0.
	v = 0.
	xpositions = Any[]
	ypositions = Any[]
	xposcar = Any[]
	yposcar = Any[]
	thetas = Any[]
	ws = Any[]
	vs = Any[]
	us = Any[]
	a_s = Any[]
	values = Any[]
	entropies = Any[]
	all_x = Any[]
	all_y = Any[]
	state = deepcopy(state_0)
	for t in 1:max_t
		thetax = state.x - env.l*sin(state.θ)
		thetay = env.l*cos(state.θ)
		push!(xpositions,[state.x,thetax])
		push!(ypositions,[0,thetay])
		push!(xposcar,[state.x])
		push!(thetas,state.θ)
		push!(ws,state.w)
		push!(vs,state.v)
		push!(us,state.u)
		if state.u == 1
			break
		end
		#ids_dstate,discretized_state = discretize_state(state,env)
		#id_dstate = build_index_b(ids_dstate,env)
		push!(values,int_value(state.θ,state.w,state.u,state.v,state.x))
		actions,policy = optimal_policy_b(state,int_value,env)
		#Choosing the action with highest prob (empowerement style)
		#idx = findmax(policy)[2] 
		#Choosing action randomly according to policy
		push!(entropies,entropy(policy))
		idx = rand(Categorical(policy))
		action = actions[idx]
		push!(a_s,action)
		#For discretized dynamics
		#_,state_p = transition_b(state,action,env) 
		#For real dynamics
		state_p = real_transition(state,action,env)
		state = deepcopy(state_p)
	#end
	end
	xpositions,ypositions,xposcar,thetas,ws,us,vs,a_s,values,entropies
end

# ╔═╡ 21472cb5-d968-4306-b65b-1b25f522dd4d
md" ## Animation"

# ╔═╡ c5c9bc66-f554-4fa8-a9f3-896875a50627
interval = collect(-0.5:0.1:0.5).*pi/180

# ╔═╡ 370a2da6-6de0-44c0-9f70-4e676769f59b
state_0_anim = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval),u=2)

# ╔═╡ 24a4ba06-d3ae-4c4b-9ab3-3852273c2fd4
function animate_w_borders(xposcar,xs,ys,xlimit,maxt,values,entropies,ip::inverted_pendulum_borders,plot_with_values = true; title = "")
	hdim = 800
	vdim = 300
	frac_for_cartpole = 0.6
	size = 1.5
	verts = [(-size,-size/2),(size,-size/2),(size,size/2),(-size,size/2)]
	anim = @animate for t in 1:length(xposcar)-1
		pcar = plot(xticks = false, yticks = false,xlim = (-xlimit-0.3,xlimit+0.3),ylim = (-0.1, ip.l + 0.05), grid = false, axis = false,legend = false)
		#Plot arena
		plot!(pcar, [-xlimit-0.2,xlimit+0.2], [-0.1,-0.1], color = :black)
		plot!(pcar, [-xlimit-0.3,-xlimit-0.2], [0.2*ip.l,0.2*ip.l], color = :black)
		plot!(pcar, [xlimit+0.2,xlimit+0.3], [0.2*ip.l,0.2*ip.l], color = :black)
		plot!(pcar, [-xlimit-0.2,-xlimit-0.2], [-0.1,0.2*ip.l], color = :black)
		plot!(pcar, [xlimit+0.2,xlimit+0.2], [-0.1,0.2*ip.l], color = :black)
		#Plot properties
		pacc = plot(ylabel = "value",xlim = (0,length(xposcar)),ylim = (round(minimum(values),sigdigits = 3)-0.1,round(maximum(values),sigdigits = 3)), xticks = false)
		pent = plot(xlabel = "time", ylabel = "action entropy",xlim = (0,length(xposcar)),ylim = (-0.1,log(ip.nactions)), xticks = false)
		if t <= maxt
			#excess = abs(xposcar[t][1]) - 1
			plot!(pcar,xs[t],ys[t],marker = (:none, 1),linewidth = 2, linecolor = :black)
			plot!(pcar,xposcar[t],[0],marker = (Shape(verts),30))
			plot!(pacc,values[1:t], label = false)
			plot!(pent, entropies[1:t], label = false)
		#scatter!(ptest,xs[t],ys[t],markersize = 50)
		#plot!(ptest,xticks = collect(0.5:env1.sizex+0.5), yticks = collect(0.5:env1.sizey+0.5), gridalpha = 0.8, showaxis = false, ylim=(0.5,env1.sizey +0.5), xlim=(0.5,env1.sizex + 0.5))
		end
		if plot_with_values == true
			plot(pcar,pacc,pent, layout = Plots.grid(3, 1, heights=[frac_for_cartpole,(1-frac_for_cartpole)/2,(1-frac_for_cartpole)/2]), size = (hdim,vdim),margin=6Plots.mm)
		else
			plot(pcar, title = title, size = (700,200),margin=5Plots.mm)
		end
	end
	anim
end

# ╔═╡ 8b38bccf-a762-439e-b19b-65e803d3c8f6
function animate_both(xposcarh,xsh,ysh,xposcarr,xsr,ysr,xlimit,maxt,ϵ,ip::inverted_pendulum_borders;with_title = true)
	hdim = 700
	vdim = 400
	size = 1.5
	verts = [(-size,-size/2),(size,-size/2),(size,size/2),(-size,size/2)]
	anim = @animate for t in 1:maxt
		pcarh = plot(xticks = false, yticks = false,xlim = (-2.2,2.2),ylim = (-0.1, ip.l + 0.05), grid = false, axis = false,legend = false)
		pcarr = plot(xticks = false, yticks = false,xlim = (-2.2,2.2),ylim = (-0.1, ip.l + 0.05), grid = false, axis = false,legend = false)
		#Plot arena
		plot!(pcarh, [-xlimit-0.2,xlimit+0.2], [-0.1,-0.1], color = :black)
		plot!(pcarh, [-xlimit-0.3,-xlimit-0.2], [0.2*ip.l,0.2*ip.l], color = :black)
		plot!(pcarh, [xlimit+0.2,xlimit+0.3], [0.2*ip.l,0.2*ip.l], color = :black)
		plot!(pcarh, [-xlimit-0.2,-xlimit-0.2], [-0.1,0.2*ip.l], color = :black)
		plot!(pcarh, [xlimit+0.2,xlimit+0.2], [-0.1,0.2*ip.l], color = :black)
		#Plot arena
		plot!(pcarr, [-xlimit-0.2,xlimit+0.2], [-0.1,-0.1], color = :black)
		plot!(pcarr, [-xlimit-0.3,-xlimit-0.2], [0.2*ip.l,0.2*ip.l], color = :black)
		plot!(pcarr, [xlimit+0.2,xlimit+0.3], [0.2*ip.l,0.2*ip.l], color = :black)
		plot!(pcarr, [-xlimit-0.2,-xlimit-0.2], [-0.1,0.2*ip.l], color = :black)
		plot!(pcarr, [xlimit+0.2,xlimit+0.2], [-0.1,0.2*ip.l], color = :black)
		if t <= maxt
			#excess = abs(xposcar[t][1]) - 1
			plot!(pcarh,xsh[t],ysh[t],marker = (:none, 1),linewidth = 2, linecolor = :black)
			plot!(pcarr,xsr[t],ysr[t],marker = (:none, 1),linewidth = 2, linecolor = :black)
			plot!(pcarh,xposcarh[t],[0],marker = (Shape(verts),30), color = "#A05A2C")
			plot!(pcarr,xposcarr[t],[0],marker = (Shape(verts),30), color = "#A05A2C")
		#scatter!(ptest,xs[t],ys[t],markersize = 50)
		#plot!(ptest,xticks = collect(0.5:env1.sizex+0.5), yticks = collect(0.5:env1.sizey+0.5), gridalpha = 0.8, showaxis = false, ylim=(0.5,env1.sizey +0.5), xlim=(0.5,env1.sizex + 0.5))
		end
		if with_title == true
			plot(pcarh,pcarr, size = (hdim,vdim),margin=5Plots.mm,layout = Plots.grid(2, 1, heigths=[0.5,0.5]), title=["H agent" "R agent, ϵ = $(ϵ)"])
		else
			plot(pcarh,pcarr, size = (hdim,vdim),margin=5Plots.mm,layout = Plots.grid(2, 1, heigths=[0.5,0.5]))
		end
	end
	anim
end

# ╔═╡ 63fe2f55-f5ce-4022-99ee-3bd4e14e6352
md"Produce animations for each? $(@bind movies CheckBox(default = false))"

# ╔═╡ 6107a0ce-6f01-4d0b-bd43-78f2125ac185
md"# R agents (reward maximizer)"

# ╔═╡ c05d12f3-8b8a-4f34-bebc-77e376a980d0
function reachable_rewards(state,action,env::inverted_pendulum_borders,δ = 1E-5)
	#We break degeneracy by adding a small cost of being far away
	r = 1 - δ*((state.θ/env.max_θ)^2+(state.x/env.max_x)^2+(state.v/env.max_v)^2+(state.w/env.max_w)^2)/4
	if state.u == 1
		r = 0
	end
	[r]
end

# ╔═╡ e181540d-4334-47c4-b35d-99023c89a2c8
function Q_iteration(env::inverted_pendulum_borders, ϵ = 0.01, tolerance = 1E-2, n_iter = 100; δ = 1E-5,verbose = false)
	v = zeros(env.nstates)
	v_new = zeros(env.nstates)
	t_stop = n_iter
	error = 0
	ferror = 0
	for t in 1:n_iter
		f_error = 0
		ferror_old = 0
		v_itp = interpolate_value(v,env)
		Threads.@threads for idx_θ in 1:env.sizeθ
		#Threads.@spawn for idx_θ in 1:env.sizeθ,idx_w in 1:env.sizew, idx_u in 1:env.sizeu, idx_v in 1:env.sizev, idx_x in 1:env.sizex
			for idx_w in 1:env.sizew, idx_u in 1:env.sizeu, idx_v in 1:env.sizev, idx_x in 1:env.sizex
				state_idx = [idx_θ,idx_w,idx_u,idx_v,idx_x]
				state = State(θ = env.θs[idx_θ], w = env.ws[idx_w], u = idx_u, v = env.vs[idx_v], x = env.xs[idx_x])
				i = build_index_b(state_idx,env)
				#v_old = deepcopy(v[i])
				#println("v before = ", v[i])
				actions,ids_actions = adm_actions_b(state,env)
				values = zeros(length(actions))
				for (id_a,a) in enumerate(actions)
					#s_primes_ids,states_p = reachable_states_b(state,a,env)
					states_p = [real_transition(state,a,env)]
					rewards = reachable_rewards(state,a,env,δ)
					#state_p = transition_b(state,a,env)[2]
					for (idx,s_p) in enumerate(states_p)
						#rewards = reachable_rewards(states_p[idx],a,env)
						#i_p = build_index_b(s_p,env)
						for r in rewards
							#deterministic environment
							values[id_a] += r + env.γ*v_itp(s_p.θ,s_p.w,s_p.u,s_p.v,s_p.x)
						end
					end
				end
				#ϵ-greedy action selection
				v_new[i] = (1-ϵ)*maximum(values) + (ϵ/length(actions))*sum(values)
				ferror = abs(v[i] - v_new[i])
				#ferror = abs(v[i] - v_old)
				ferror_old = max(ferror_old,ferror)
			end
		end
		error = norm(v-v_new)/norm(v)
		#if f_error < tolerance
		if ferror_old < tolerance
			t_stop = t
			break
		end
		if verbose == true
			println("iteration = ", t, ", error = ", error, ", max function error = ", ferror_old)
		end
		v = deepcopy(v_new)
	end
	v,error,t_stop
end

# ╔═╡ 31a1cdc7-2491-42c1-9988-63650dfaa3e3
function optimal_policy_q(state,value,ϵ,env::inverted_pendulum_borders,interpolation = true)
	actions,ids_actions = adm_actions_b(state,env)
	values = zeros(length(actions))
	policy = zeros(length(actions))
	#print("actions = ", actions)
	#state_index = build_nonflat_index_b(state,env)
	#id_state = build_index_b(state_index,env)
	for (idx,a) in enumerate(actions)
		#s_primes_ids,states_p = reachable_states_b(state,a,env)
		s_prime = real_transition(state,a,env)
		rewards = reachable_rewards(state,a,env)
		#for (id_sp,s_p) in enumerate(s_primes_ids)
			for r in rewards
				#if interpolation == true
					#interpolated value
					values[idx] += r + env.γ*value(s_prime.θ,s_prime.w,s_prime.u,s_prime.v,s_prime.x)
				#else
					#deterministic environment
					#i_p = build_index_b(s_primes_ids[1],env)
					#values[idx] += r + env.γ*value[i_p]
				#end
			end
		#end
	end
	#println("state = ", state)
	#println("policy = ", policy)
	#println("sum policy = ", sum(policy))
	#println("values = ", values)
	best_actions = findall(i-> i == maximum(values),values)
	#ϵ-greedy policy
	for i in 1:length(actions)
		if i in best_actions
			#There might be more than one optimal action
			policy[i] = (1-ϵ)/length(best_actions) + ϵ/length(actions)
		else
			#Choose random action with probability ϵ
			policy[i] = ϵ/length(actions)
		end
	end
	actions,policy,length(best_actions)
end

# ╔═╡ 87c0c3cb-7059-42ae-aed8-98a0ef2eb55f
ip_q = ip_b #same enviromment than H agent
#inverted_pendulum_borders(M = 1.0, m = 0.1,l = 1.,Δt = 0.02, sizeθ = ip_b.sizex, sizew = ip_b.sizew,sizev = ip_b.sizev, sizex = ip_b.sizex, a_s = [-50,-10,0,10,50], max_θ = ip_b.max_θ, max_x = 2.4, max_v = 3, max_w = 3, nactions = 5, γ = ip_b.γ)

# ╔═╡ f9b03b4d-c521-4456-b0b9-d4a301d8a813
md" ## Value iteration"

# ╔═╡ 355db008-9661-4e54-acd5-7c2c9ba3c7f5
tol = 1E-2

# ╔═╡ 8a6b67d0-8008-4a4c-be7c-0c0b76311385
ϵ_try = 0.0

# ╔═╡ c2105bee-c29d-4853-9388-31c405283395
#not important 
δ_r = 0.0

# ╔═╡ 564cbc7a-3125-4b67-843c-f4c74ccef51f
begin
#To calculate value, uncomment, it takes around 30 minutes for 1.8E6 states
#q_value, q_error, q_stop = Q_iteration(ip_q,ϵ_try,tol,1200,δ = δ_r,verbose = false)
#Otherwise, read from file
#Read from compressed file
	#q_zip = ZipFile.Reader("q_value_g_$(ip_q.γ)_nstates_$(ip_q.nstates).dat.zip")
	#q_value = readdlm(q_zip.files[1], Float64)
	#q_value = readdlm("values_borders/q_value_eps_$(ϵ_try)_nstates_$(ip_q.nstates)_xlim_2.4.dat");
	q_value = readdlm("values/q_value_g_$(ip_b.γ)_eps_$(ϵ_try)_nstates_$(ip_q.nstates).dat")
end

# ╔═╡ 288aa06b-e07b-41cc-a51f-49f780c634b8
begin
	b = 2.4
	ip_border = ip_b #inverted_pendulum_borders(M = 1, m = 0.1,l = 1.,Δt = 0.02, sizeθ = 31, sizew = 31,sizev = 31, sizex = 31, max_θ = 0.62, max_w = 3, a_s = [-50,-10,0,10,50], max_x = b, max_v = 3, nactions = 5, γ = 0.96)
	#q_val = readdlm("values_borders/q_value_eps_$(ϵ_try)_nstates_$(ip_border.nstates)_xlim_$(b).dat")
	#h_val = readdlm("values_borders/h_value_nstates_$(ip_border.nstates)_xlim_$(b).dat")
	q_val = q_value
	h_val = h_value
	q_val_int = interpolate_value(q_val,ip_border)
	h_val_int = interpolate_value(h_val,ip_border)
end;

# ╔═╡ 9230de54-3ee3-4242-bc34-25a38edfbb6b
begin
	max_t_h_anim = 1000
	#state0_h_anim = State(θ = 0.001, u = 2)
	xs_h_anim, ys_h_anim, xposcar_h_anim, thetas_ep_h_anim, ws_ep_h_anim, us_ep_h_anim, vs_ep_h_anim, actions_h_anim, values_h_anim, entropies_h_anim = create_episode_b(state_0_anim,h_val_int,max_t_h_anim, ip_border)
	length(xposcar_h_anim)
end

# ╔═╡ dafbc66b-cbc2-41c9-9a42-da5961d2eaa6
@bind t Slider(1:max_t_h_anim)

# ╔═╡ 2735cbf5-790d-40b4-ac32-a413bc1d530a
begin
	draw_cartpole(xposcar_h_anim,xs_h_anim,ys_h_anim,t,ip_b.max_x,ip_b)
	#savefig("cartpole_draw.pdf")
end

# ╔═╡ 2f1b6992-3082-412d-b966-2b4b278b2ed0
draw_cartpole_time(xposcar_h_anim,xs_h_anim,ys_h_anim,ip_b.max_x,200,1,ip_b)

# ╔═╡ a54788b9-4b1e-4066-b963-d04008bcc242
begin
	draw_cartpole_timeright(xposcar_h_anim,xs_h_anim,ys_h_anim,800,1,ip_b)
	#savefig("h_frozen_movie2.pdf")
end

# ╔═╡ 14314bcc-661a-4e84-886d-20c89c07a28e
#Animation, it takes some time
if movies == true
	anim_b = animate_w_borders(xposcar_h_anim,xs_h_anim,ys_h_anim,ip_b.max_x,max_t_h_anim,values_h_anim,entropies_h_anim,ip_b,false)
end

# ╔═╡ c087392c-c411-4368-afcc-f9a104856884
if movies == true
gif(anim_b,fps = Int(1/ip_b.Δt))#,"episode_h_agent.gif")
end

# ╔═╡ d20c1afe-6d5b-49bf-a0f2-a1bbb21c709f
#If calculated, this line writes the value function in a file
#writedlm("values/q_value_eps_$(ϵ_try)_nstates_$(ip_q.nstates).dat",q_value)

# ╔═╡ e9687e3f-be56-44eb-af4d-f169558de0fd
q_value_int = interpolate_value(q_value,ip_q);

# ╔═╡ e9a01524-90b1-4249-a51c-ec8d1624be5b
function create_episode_q(state_0,value,ϵ,max_t,env::inverted_pendulum_borders,interpolation = true)
	x = 0.
	v = 0.
	xpositions = Any[]
	ypositions = Any[]
	xposcar = Any[]
	yposcar = Any[]
	thetas = Any[]
	ws = Any[]
	vs = Any[]
	us = Any[]
	a_s = Any[]
	values = Any[]
	entropies = Any[]
	rewards = Any[]
	all_x = Any[]
	all_y = Any[]
	state = deepcopy(state_0)
	for t in 1:max_t
		thetax_new = state.x - env.l*sin(state.θ)
		thetay_new = env.l*cos(state.θ)
		push!(xpositions,[state.x,thetax_new])
		push!(ypositions,[0,thetay_new])
		push!(xposcar,[state.x])
		push!(thetas,state.θ)
		push!(ws,state.w)
		push!(vs,state.v)
		push!(us,state.u)
		#actions_at_s,_ = adm_actions_b(state,env)
		# policy = ones(length(actions))./length(actions)
		#ids_dstate,discretized_state = discretize_state(state,env)
		#id_dstate = build_index_b(ids_dstate,env)
		#if interpolation == true
			push!(values,value(state.θ,state.w,state.u,state.v,state.x))
		#else
		#	push!(values,value[id_dstate])
		#end
		actions_at_s,policy,n_best_actions = optimal_policy_q(state,value,ϵ,env)
		#Choosing action randomly from optimal action set according to policy
		#action = rand(actions)
		#ϵ-greedy: with prob ϵ choose a random action from action set
		# if rand() < ϵ
		# 	action = rand(actions_at_s)
		# end
		idx = rand(Categorical(policy))
		action = actions_at_s[idx]
		#There might be degeneracy in optimal action, so choose with (1-ϵ)
		prob_distribution = ones(n_best_actions).*((1-ϵ)/n_best_actions)
		#Then, choose with ϵ between available actions
		for i in 1:length(actions_at_s)
			push!(prob_distribution,ϵ/length(actions_at_s))
		end
		#Compute entropy of that distribution
		push!(entropies,entropy(prob_distribution))
		#idx = rand(Categorical(policy))
		#action = actions[idx]
		push!(a_s,action)
		r = reachable_rewards(state,action,env)[1]
		push!(rewards,r)

		#For discretized dynamics
		#_,state_p = transition_b(state,action,env) 
		#For real dynamics
		if state.u == 1
			break
		end
		state_p = real_transition(state,action,env)
		state = deepcopy(state_p)
	#end
	end
	xpositions,ypositions,xposcar,thetas,ws,us,vs,a_s,values,entropies,rewards
end

# ╔═╡ 4893bf14-446c-46a7-b695-53bac123ff99
md" ## Animation"

# ╔═╡ 7edf8ddf-2b6e-4a4b-8181-6b8bbdd22841
begin
	max_t_q_anim = max_t_h_anim#1500
	state0_q_anim = state_0_anim#State(θ = 0,x = 0, v = 0,w = 0, u = 2) #state_0_anim
	ϵ_anim = 0.0 #ϵ_try
	xs_q_anim, ys_q_anim, xposcar_q_anim, thetas_ep_q_anim, ws_ep_q_anim, us_ep_q_anim, vs_ep_q_anim, actions_q_anim, values_q_anim, entropies_q_anim,rewards_q_anim = create_episode_q(state0_q_anim,q_val_int,ϵ_anim,max_t_q_anim, ip_border, interpolation)
	length(xposcar_q_anim)
end

# ╔═╡ aa4229d3-e221-4a87-8a18-ed376d33d3ac
draw_cartpole_time(xposcar_q_anim,xs_q_anim,ys_q_anim,ip_b.max_x,200,1,ip_b)

# ╔═╡ 69128c7a-ddcb-4535-98af-24fc91ec0b7e
begin
	draw_cartpole_timeright(xposcar_q_anim,xs_q_anim,ys_q_anim,800,1,ip_b)
	#savefig("q_frozen_movie_eps0.0.pdf")
end

# ╔═╡ c98025f8-f942-498b-9368-5d524b141c62
if movies == true
 	anim_q = animate_w_borders(xposcar_q_anim,xs_q_anim,ys_q_anim,ip_q.max_x,max_t_q_anim,values_q_anim,entropies_q_anim,ip_q,true)
end

# ╔═╡ d8642b83-e824-429e-ac3e-70e875a47d1a
if movies == true
gif(anim_q,fps = Int(round(1/ip_q.Δt)),"episode_q_agent_epsilon_$(ϵ_anim)_g_$(ip_q.γ)_wvalues.gif")
end

# ╔═╡ 90aff8bb-ed69-40d3-a22f-112f713f4b93
md"# Comparison between H and R agents"

# ╔═╡ 85e789c6-8322-4264-ab70-dc33d64c4de4
md"Produce animation for both? $(@bind movie_both CheckBox(default = false))"

# ╔═╡ 4fad0692-05dc-4c3b-9aae-cd9a43519e51
md"## ϵ-greedy policy survival rate analysis"

# ╔═╡ 973b56e5-ef3c-4328-ad26-3ab63650537e
ϵs_totry = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.33,0.35,0.4]

# ╔═╡ 3d567640-78d6-4f76-b13e-95be6d5e9c64
# begin
# 	#Calculate q values for several ϵs
# 	ϵs_to_compute_value = [0.22] #ϵs_totry #[0.25,0.3,0.35,0.4]
# 	for (i,ϵ) in enumerate(ϵs_to_compute_value)
# 		println("epsilon = ", ϵ)
# 		q_val, _, _ = Q_iteration(ip_q,ϵ,tol,1200,δ = δ_r)
# 		writedlm("values/q_value_g_$(ip_q.γ)_eps_$(ϵ)_nstates_$(ip_q.nstates).dat",q_val)
# 	end
# end

# ╔═╡ 06f064cf-fc0d-4f65-bd6b-ddb6e4154f6c
begin
	max_time = 10000
	num_episodes = 1000
	ϵs = [0.0,0.05,0.1,0.15,0.2,0.22,0.25,0.3] #ϵs_totry# [0.0,0.001,0.01,0.05] 
	# #To compute the survival times for various epsilon-greedy Q agents, it takes a bit less than 10 min
	# survival_pcts = zeros(length(ϵs),num_episodes)
	# for (i,ϵ) in enumerate(ϵs)
	# 	q_val = readdlm("values/q_value_g_$(ip_b.γ)_eps_$(ϵ)_nstates_$(ip_q.nstates).dat")
	# 	q_val_int = interpolate_value(q_val,ip_q)
	# 	Threads.@threads for j in 1:num_episodes
	# 		#println("j = ", j)
	# 		state0 = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval), u = 2)
	# 		xs_q, ys_q, xposcar_q, thetas_ep_q, ws_ep_q, us_ep_q, vs_ep_q, actions_q = create_episode_q(state0, q_val_int, ϵ, max_time, ip_q, interpolation)
	# 		#survival_timesq[j] = length(xposcar_q)
	# 		#if length(xposcar_q) == max_time
	# 			survival_pcts[i,j] = length(xposcar_q)
	# 		#end
	# 	end
	# end
	# writedlm("survivals/survival_pcts_R_g_$(ip_b.γ)_epsilon_maxtime_$(max_time)_nstates_$(ip_b.nstates).dat",survival_pcts)
	
	# # #Computes it for H agent
	# survival_H = zeros(num_episodes)
	# Threads.@threads for i in 1:num_episodes
	# 	state0_b = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval), u = 2)
	# 	xs_b, ys_b, xposcar_b, thetas_ep_b, ws_ep_b, us_ep_b, vs_ep_b, actions_b = create_episode_b(state0_b,h_value_int,max_time, ip_b)
	# 	survival_H[i] = length(xposcar_b)
	# end
	# writedlm("survivals/survival_pcts_H_g_$(ip_b.γ)_nstates_$(ip_b.nstates).dat",survival_H)
	
	# #Otherwise, read from file
	survival_pcts = readdlm("survivals/survival_pcts_R_g_$(ip_b.γ)_epsilon_maxtime_$(max_time)_nstates_$(ip_b.nstates).dat")
	 survival_H = readdlm("survivals/survival_pcts_H_g_$(ip_b.γ)_nstates_$(ip_b.nstates).dat")
end;

# ╔═╡ ac43808a-ef2a-472d-9a9e-c77257aaa535
survival_Q = survival_pcts

# ╔═╡ da37a5c8-b8ef-4131-a153-50c21461d9c4
# begin
# 	survival_Q_extra = zeros(5,10000)
# 	survival_Q_extra[1:3,:] = survival_Q[1:3,:]
# 	survival_Q_extra[4,:] = survival_pcts
# 	survival_Q_extra[5,:] = survival_Q[4,:]
# end

# ╔═╡ a8b1a61c-6bd0-417b-8d58-8c8f8d77da7e
#writedlm("survival_pcts_Q_agent.dat",survival_Q_extra)

# ╔═╡ 1a6c956e-38cc-4217-aff3-f303af0282a4
md"Density plot? $(@bind density CheckBox(default = false))"

# ╔═╡ e169a987-849f-4cc2-96bc-39f234742d93
begin
	bd = 2000
	surv_hists = plot(xlabel = "Survived time steps", xticks = [10000,50000,100000])
	if density == true
		plot!(surv_hists,ylabel = "Density")
		density!(surv_hists, bandwidth = bd, survival_H,label = "H agent",linewidth = 2)
	else
		plot!(surv_hists,ylabel = "Normalized frequency")
		plot!(surv_hists,bins = collect(-bd/2:bd:max_time+bd/2),survival_H,st = :stephist, label = "H agent", alpha = 1.0,linewidth = 2,normalized = :probability)
	end
	alphas = ones(length(ϵs))
	for i in 1:length(ϵs)
		if density == true
			density!(surv_hists, bandwidth = bd, survival_Q[i,:],label = "ϵ = $(ϵs[i])",linewidth = 2)
		else
			plot!(surv_hists,bins = (collect(-bd/2:bd:max_time+bd/2)),survival_Q[i,:],st = :stephist,normalized = :probability,label = "ϵ = $(ϵs[i])",alpha = alphas[i],linewidth = 2)
		end
	end
	plot(surv_hists, grid = true, minorgrid = false, legend_position = :top,margin = 4Plots.mm,size = (500,500))
	#savefig("q_h_survival_histograms.pdf")
end

# ╔═╡ ae15e079-231e-4ea2-aabb-ee7d44266c6d
begin
	surv_means = plot(xlabel = "ϵ")
	plot!(surv_means,ylabel = "Survived time steps",yticks = [9800,10000])
	plot!(surv_means, ϵs, mean(survival_H).*ones(length(ϵs)),label = "H agent",linewidth = 2.5, color = "blue")#,yerror = std(survival_H./max_time)./sqrt(length(survival_H)))
	plot!(surv_means,ϵs,mean(survival_Q,dims = 2),yerror = std(survival_Q,dims = 2)./(sqrt(length(survival_Q[1,:]))),markerstrokewidth = 2, linewidth = 2.5,label = "R agent",color = "orange")
	plot(surv_means, grid = false, legend_position = :bottomleft,margin = 2Plots.mm,size = (450,300),bg_legend = :white,fg_legend = :white,fgguide = :black)
	#savefig("q_h_survival_epsilon.pdf")
end

# ╔═╡ a4b26f44-319d-4b90-8fee-a3ab2418dc47
md"## State occupancy histograms"

# ╔═╡ 94da3cc0-6763-40b9-8773-f2a1a2cbe507
state_0_comp = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval),u=2)

# ╔═╡ 69b6d1f1-c46c-43fa-a21d-b1f61fcd7c55
time_histograms = 30000

# ╔═╡ 7a27d480-a9cb-4c26-91cd-bf519e8b35fa
begin
	max_t_b = time_histograms
	state0_b = state_0_comp
	xs_b, ys_b, xposcar_b, thetas_ep_b, ws_ep_b, us_ep_b, vs_ep_b, actions_b, values_b, entropies_b = create_episode_b(state0_b,h_value_int,max_t_b, ip_b)
	#Check if it survived the whole episode
	length(xposcar_b)
end

# ╔═╡ 2c4375df-5063-43ea-9578-69ec94834362
ip_b

# ╔═╡ 339a47a1-2b8a-4dc2-9adc-530c53d66fb1
#To check that cartpole does not exceed maximum speed in discretize state space
length([vs_ep_b[i] for i in 1:length(vs_ep_b) if vs_ep_b[i]== 6])

# ╔═╡ 8c819d68-3981-4073-b58f-8fde5b73be33
#To check that cartpole does not exceed maximum angular speed in discretized state space
length([ws_ep_b[i] for i in 1:length(ws_ep_b) if ws_ep_b[i]==3])

# ╔═╡ b367ccc6-934f-4b18-b1db-05286111958f
begin
	max_t_q = time_histograms
	state0_q = state_0_comp
	ϵ = 0.25# ϵ_try
	q_value_hist = readdlm("values/q_value_g_$(ip_b.γ)_eps_$(ϵ)_nstates_$(ip_q.nstates).dat")
	q_value_int_hist = interpolate_value(q_value_hist,ip_q)
	xs_q, ys_q, xposcar_q, thetas_ep_q, ws_ep_q, us_ep_q, vs_ep_q, actions_q, values_q, entropies_q,rewards_q = create_episode_q(state0_q,q_value_int_hist,ϵ,max_t_q, ip_q, interpolation)
	#Check if it survived the whole episode
	length(xposcar_q)
end

# ╔═╡ 43ce27d1-3246-4045-b3aa-a99bcf25cbaa
if movie_both == true
	maxt = max_t_q_anim #min(length(xposcar_h_anim),length(xposcar_q_anim))
	println("H agent lived $(length(xposcar_h_anim)), and R agent lived $(length(xposcar_q_anim))")
	anim_both = animate_both(xposcar_b,xs_b,ys_b,xposcar_q,xs_q,ys_q,ip_border.max_x,maxt,ϵ,ip_border,with_title = false)
end

# ╔═╡ 0832d5e6-7e92-430f-afe3-ddb5e55dc591
gif(anim_both,fps = Int(1/ip_border.Δt),"dummy.gif")

# ╔═╡ 096a58e3-f417-446e-83f0-84a333880680
begin
	x_b = Any[]
	x_q = Any[]
	for i in 1:length(xposcar_b)
	 push!(x_b,xposcar_b[i][1])
	end
	for i in 1:length(xposcar_q)
	 push!(x_q,xposcar_q[i][1])
	end
end

# ╔═╡ c95dc4f2-1f54-4266-bf23-7d24cee7b3d4
begin
	cbarlim = 0.003
	p1h = plot(ylim = (-30,30),xlim=(-2,2),colorbarticks = [0.0,0.001,0.002])#,xlabel = "Position \$x\$", ylabel = "Angle \$\\theta\$",title = "H agent")
	#heatmap!(p1h,zeros(160,160))
	plot!(p1h,x_b,thetas_ep_b.*180/pi,bins = (70,70),st = :histogram2d,normed = :probability,clim = (0,cbarlim),cbar = false)
	p2h = plot(ylim = (-30,30),xlim=(-2,2))#,xlabel = "Position \$x\$", title = "R agent, ϵ = $(ϵ)")
	plot!(p2h,x_q,thetas_ep_q.*180/pi,bins = (70,70),st = :histogram2d,normed = :probability,clim = (0,cbarlim))
	plot(p1h,p2h,size = (700,300),layout = Plots.grid(1, 2, widths=[0.44,0.56]),margin = 6Plots.mm,grid = false)
	#savefig("angle_position_histogram_eps_$(ϵ).pdf")
end

# ╔═╡ d2139819-83a1-4e2b-a605-91d1e686b9a3
begin
	cbarlim_v = 0.005
	p1h_v = plot(xlim = (-ip_b.max_w*180/pi,ip_b.max_w*180/pi),ylim=(-ip_b.max_v,ip_b.max_v),xlabel = "Angular speed", ylabel = "Speed")
	#heatmap!(p1h,zeros(160,160))
	plot!(p1h_v,ws_ep_b.*180/pi,vs_ep_b,bins = (40,40),st = :histogram2d,normed = :probability,margin = 5Plots.mm,clim = (0,cbarlim_v),cbar = false)
	p2h_v = plot(xlim = (-ip_b.max_w*180/pi,ip_b.max_w*180/pi),ylim=(-ip_b.max_v,ip_b.max_v),xlabel = "Angular speed")
	plot!(p2h_v,ws_ep_q.*180/pi,vs_ep_q,bins = (40,40),st = :histogram2d,normed = true,margin = 5Plots.mm,clim = (0,cbarlim_v))
	plot(p1h_v,p2h_v,size = (1000,500))
end

# ╔═╡ fa9ae665-465d-4f3e-b8cd-002c80420adb
begin
	cbarlim2 = 0.003
	p1h_p = plot(xlim = (-36,36),ylim=(-ip_b.max_w*180/pi,ip_b.max_w*180/pi),xlabel = "Angle", ylabel = "Angular speed")
	#heatmap!(p1h,zeros(160,160))
	plot!(p1h_p,thetas_ep_b.*180/pi,ws_ep_b.*180/pi,bins = (40,40),st = :histogram2d,normed = :probability,margin = 5Plots.mm,clim = (0,cbarlim2),cbar = false)
	p2h_p = plot(xlim = (-36,36),ylim=(-ip_b.max_w*180/pi,ip_b.max_w*180/pi),xlabel = "Angle")
	plot!(p2h_p,thetas_ep_q.*180/pi,ws_ep_q.*180/pi,bins = (40,40),st = :histogram2d,normed = :probability,margin = 5Plots.mm,clim = (0,cbarlim2))
	plot(p1h_p,p2h_p,size = (1000,500))
end

# ╔═╡ 6cc17343-c725-4653-ad48-b3535a53b09e
begin
	cbarlim3 = 0.003
	p1h_p2 = plot(xlim = (-1.5,1.5),ylim=(-ip_b.max_v,ip_b.max_v),xlabel = "Position", ylabel = "Speed")#, title = "H agent")
	#heatmap!(p1h,zeros(160,160))
	plot!(p1h_p2,x_b,vs_ep_b,bins = (40,40),st = :histogram2d,normed = :probability,clim = (0,cbarlim3),cbar = false)
	p2h_p2 = plot(xlim = (-1.5,1.5),ylim=(-ip_b.max_v,ip_b.max_v),xlabel = "Position")#, title = "R agent")
	plot!(p2h_p2,x_q,vs_ep_q,bins = (40,40),st = :histogram2d,normed = :probability,clim = (0,cbarlim3))
	plot(p1h_p2,p2h_p2,size = (900,400),layout = Plots.grid(1, 2, widths=[0.45,0.55]),margin = 7Plots.mm)
	#savefig("speed_position_histogram.svg")
end

# ╔═╡ 9fa87325-4af3-4ec0-a716-80652dcf2ace
md"Angle vs Position? $(@bind angle CheckBox(default = true))"

# ╔═╡ 1aace394-c7da-421a-bd4c-e7c7d8b36231
t_trajectories = 2000

# ╔═╡ c9e4bc33-046f-4ebd-9da7-a768886107ef
if angle
	limy = ip_b.max_θ*180/pi
	labely = "Angle"
else
	limy = ip_b.max_v
	labely = "Speed"
end

# ╔═╡ f4b1a0bf-aff3-4b49-b15f-e5fdf31d969c
begin
	phase_space_trajectory = plot(1,xlabel = "Position",ylabel = labely,xlim = (-2.,2.),ylim=(-limy,limy),label = false, title = "R agent, ϵ = $(ϵ)",marker = (:black,2))
	@gif for i in 1:t_trajectories#length(x_b)
		if angle
			push!(phase_space_trajectory,x_q[i],thetas_ep_q[i]*180/pi)
		else
			push!(phase_space_trajectory,x_q[i],vs_ep_q[i])
		end
	end every 10
	#savefig("path_R_agent.gif")
end

# ╔═╡ bde8e418-9391-4c5a-a754-5ae1e3215e66
begin
	ps_trajectory = plot(1,xlabel = "Position",ylabel = labely,xlim = (-2.,2.),ylim=(-limy,limy),label = false, title = "H agent \$\\beta = 0.0\$",marker = (:black,2))
	@gif for i in 1:t_trajectories#length(x_b)
		if angle
			push!(ps_trajectory,x_b[i],thetas_ep_b[i]*180/pi)
		else
			push!(ps_trajectory,x_b[i],vs_ep_b[i])
		end
	end every 10
end

# ╔═╡ ac2816ee-c066-4063-ae74-0a143df37a9c
begin
	ps_h = plot(1,xlabel = "Position",ylabel = labely,xlim = (-2.,2.),ylim=(-limy,limy),label = false, title = "H agent",marker = (:black,1.5),lc = "blue",lw = 2)
	ps_r = plot(1,xlabel = "Position",ylabel = labely,xlim = (-2.,2.),ylim=(-limy,limy),label = false, title = "R agent, ϵ = $(ϵ)",marker = (:black,1.5),lc = "orange",lw = 2)
	colgrad = cgrad(:roma)
	@gif for i in 1:2:t_trajectories#length(x_b)
		if angle
			push!(ps_h,x_b[i],thetas_ep_b[i]*180/pi)
			push!(ps_r,x_q[i],thetas_ep_q[i]*180/pi)
			plot!(ps_h)
		else
			push!(ps_r,x_q[i],vs_ep_q[i])
			push!(ps_h,x_b[i],vs_ep_b[i])
		end
		#plot!(ps_h)
		plot(ps_h,ps_r,size = (800,400),margin = 5Plots.mm)
	end every 5
end

# ╔═╡ d0c487cb-041f-4c8d-9054-e5f3cfad1ed4
# begin
# 	#title = plot(title = "Initial state: θ = $(round(state_0_comp.θ*180/pi,sigdigits=2)) deg, ω = $(round(state_0_comp.w*180/pi,sigdigits=2)) deg/s, x = $(round(state_0_comp.x,sigdigits=2)) m,  v = $(round(state_0_comp.v,sigdigits=2)) m/s", grid = false, axis = false, ticks = false, bottom_margin = -20Plots.px)
# 	p1 = plot(xlim=(-36,36), xlabel = "Angle (deg)")
# 	plot!(p1,thetas_ep_b.*180/pi,st = :density,label = false,lw = 2)
# 	plot!(p1,thetas_ep_q.*180/pi,st = :density,label = false,lw = 2)
# 	plot!(p1,ylabel = "Density")
# 	p2 = plot(xlim=(-3.5*180/pi,3.5*180/pi), xlabel = "Angular speed (deg/s)")
# 	plot!(p2,ws_ep_b.*180/pi,st = :density,label = "h agent",lw = 2)
# 	plot!(p2,ws_ep_q.*180/pi,st = :density,label = "q agent, \nϵ = $(ϵ)",lw = 2)
# 	plot!(p2, legend_foreground_color = nothing,legend_position = :topright)
# 	p3 = plot(xlim=(-2.,2.))
# 	#plot!(p3,bins = collect(-2.05:0.1:2.05),x_b,st = :stephist)
# 	#plot!(p3,bins = collect(-2.05:0.1:2.05),x_q,st = :stephist)
# 	plot!(p3,x_b,st = :density, label = false, xlabel = "Position",lw = 2)
# 	plot!(p3,x_q,st = :density,label = false,lw = 2)
# 	plot!(p3,ylabel = "Density")
# 	p4 = plot(xlim=(-4,4))
# 	plot!(p4,vs_ep_b,st = :density, bandwidth = 0.1,label = false, xlabel = "Linear speed",lw = 2)
# 	plot!(p4,vs_ep_q,st = :density, bandwidth = 0.1,label = false,lw = 2)
# 	plot!(p4)
# 	#plot(title,p1,p2,p3,p4, size = (900,500), layout = @layout([A{0.01h}; [B C]; [D E]]))
# 	plot(p1,p2,p3,p4, size = (900,600), margin = 4Plots.mm)
# 	#savefig("histograms_epsilon_$(ϵ).pdf")
# end

# ╔═╡ e25f575e-91d5-479e-a752-a831a0692f26
md"# Stochastic half arena"

# ╔═╡ 0e7f28f3-53b2-431d-afd9-d2fe6c511863
@with_kw struct pars
	α = 1
	β = 1
	#probability of not performing your action (performing "nothing" instead)
	η = 0.0
end

# ╔═╡ e0c1c42b-4327-4ad8-b097-92bf08912e3e
η_test = 0.4

# ╔═╡ 47c63b41-4385-479f-b0e9-afae0ed08058
params = pars(β = 1.0,η = η_test)

# ╔═╡ d3cbbcca-b43a-421e-b29f-4388a409de41
function prob_transition(state::State,action,params,env::inverted_pendulum_borders)
	acc = dwdt(state.θ,state.w,action,env)
	acc_0 = dwdt(state.θ,state.w,0,env)
	new_th = state.θ + state.w*env.Δt #+ acc*env.Δt^2/2
	new_w = state.w + acc*env.Δt
	new_w_0 = state.w + acc_0*env.Δt
	#According to the paper, but there is a sign error
	#acc_x = env.α*(action + env.m*env.l*(state.w^2*sin(state.θ)-acc*cos(state.θ)))
	acc_x = (4/3 * env.l * acc - env.g*sin(state.θ))/cos(state.θ)
	acc_x_0 = (4/3 * env.l * acc_0 - env.g*sin(state.θ))/cos(state.θ)
	new_v = state.v + acc_x*env.Δt
	new_v_0 = state.v + acc_x_0*env.Δt
	new_x = state.x + state.v*env.Δt #+ acc_x*env.Δt^2/2
	new_u = state.u
	if abs(new_th) >= env.max_θ 
		new_th = sign(new_th)*env.max_θ
		new_u = 1
	end
	if abs(new_x) >= env.max_x
		new_x = sign(new_x)*env.max_x
		new_u = 1
	end
	if abs(new_v) >=env.max_v
		new_v = sign(new_v)*env.max_v
	end
	if abs(new_v_0) >=env.max_v
		new_v_0 = sign(new_v_0)*env.max_v
	end
	if abs(new_w_0) >=env.max_w
		new_w_0 = sign(new_w_0)*env.max_w
	end
	if abs(new_w) >= env.max_w
		new_w = sign(new_w)*env.max_w
	end
	#You can fail at using your action only at x>0 part of the arena
	if state.x > 0
		prob = params.η
	else
		prob = 0
	end
	[State(θ = new_th, w = new_w, v = new_v, x = new_x, u = new_u),State(θ = new_th, w = new_w_0, v = new_v_0, x = new_x, u = new_u)],[1-prob,prob]
end

# ╔═╡ bcff0238-4182-4407-a8b7-f19e6b700906
function h_iteration_half(params,env::inverted_pendulum_borders; tolerance = 1E0, n_iter = 100,verbose = false)
	v = zeros(env.nstates)
	v_new = zeros(env.nstates)
	t_stop = n_iter
	error = 0
	f_error = 0
	for t in 1:n_iter
		#println("iter = ",t)
		v_itp = interpolate_value(v,env)
		ferror_max = 0
		#Parallelization over states
		Threads.@threads for idx_θ in 1:env.sizeθ
			for idx_w in 1:env.sizew, idx_u in 1:env.sizeu, idx_v in 1:env.sizev, idx_x in 1:env.sizex
				
				state_idx = [idx_θ,idx_w,idx_u,idx_v,idx_x]
				state = State(θ = env.θs[idx_θ], w = env.ws[idx_w], u = idx_u, v = 
				env.vs[idx_v], x = env.xs[idx_x])
				#println("state = ", state)
				i = build_index_b(state_idx,env)
				actions,ids_actions = adm_actions_b(state,env)
				sum = 0
				# Add negligible external reward that breaks degeneracy for 
				# Q agent
				#small_reward = degeneracy_cost(state,env,params.δ)
				for (id_a,a) in enumerate(actions)
					#For every action, look at reachable states
					#s_primes_ids,states_p = reachable_states_b(state,a,env)
					states_p,probs_p = prob_transition(state,a,params,env)
					#states_p = [real_transition(state,a,env)]
					exponent = 0
					for (idx,s_p) in enumerate(states_p)
						#exponent += env.γ*1*v_itp(s_p.θ,s_p.w,s_p.u,s_p.v,s_p.x)
						if probs_p[idx] > 0
							exponent += -params.β*probs_p[idx]*log(probs_p[idx]) + env.γ*probs_p[idx]*v_itp(s_p.θ,s_p.w,s_p.u,s_p.v,s_p.x)
						end
					end
					sum += exp(exponent/params.α)# + small_reward)
				end
				v_new[i] = params.α*log(sum)
				f_error = abs(v[i] - v_new[i])
				# Use supremum norm of difference between values
				# at different iterations
				ferror_max = max(ferror_max,f_error)
			end
		end
		# Check overall error between value's values at different iterations
		error = norm(v-v_new)/norm(v)
		#if f_error < tolerance
		if ferror_max < tolerance
			t_stop = t
			break
		end
		if verbose == true
		println("iteration = ", t, ", error = ", error, ", max function error = ", ferror_max)
		end
		v = deepcopy(v_new)
	end
	v_new,error,t_stop
end

# ╔═╡ 581515a9-3d39-44ac-be92-e3049a36a15d
function optimal_policy_half(state,value,params,env::inverted_pendulum_borders)
	#Check admissible actions
	actions,ids_actions = adm_actions_b(state,env)
	policy = zeros(length(actions))
	#state_index = build_nonflat_index_b(state,env)
	#id_state = build_index_b(state_index,env)
	#Value at current state
	v_state = value(state.θ,state.w,state.u,state.v,state.x)
	for (idx,a) in enumerate(actions)
		#Given action, check reachable states (deterministic environment for now)
		#s_primes_ids,states_p = reachable_states_b(state,a,env)
		s_primes,probs_p = prob_transition(state,a,params,env)
		exponent = 0
		for (idx_p,s_p) in enumerate(s_primes)
			#P = 1
			#if interpolation == true
				#interpolated value
				if probs_p[idx_p] > 0 
					exponent += -params.β*probs_p[idx_p]*log(probs_p[idx_p]) + env.γ*probs_p[idx_p]*value(s_p.θ,s_p.w,s_p.u,s_p.v,s_p.x)
				end
			# else
			# 	#deterministic environment
			# 	i_p = build_index_b(s_primes_ids[1],env)
			# 	exponent += env.γ*P*value[i_p]
			# end
		end
		#Normalize at exponent
		exponent -= v_state
		policy[idx] = exp(exponent/params.α)
	end
	#Since we are using interpolated values, policy might not be normalized, so we normalize
	#println("sum policy = ", sum(policy))
	policy = policy./sum(policy)
	#Return available actions and policy distribution over actions
	actions,policy
end

# ╔═╡ 5a71d63b-a63e-4ad7-abc4-36c2f2c61711
function create_episode_half(state_0,int_value,max_t,params,env::inverted_pendulum_borders)
	x = 0.
	v = 0.
	xpositions = Any[]
	ypositions = Any[]
	xposcar = Any[]
	yposcar = Any[]
	thetas = Any[]
	ws = Any[]
	vs = Any[]
	us = Any[]
	a_s = Any[]
	values = Any[]
	entropies = Any[]
	all_x = Any[]
	all_y = Any[]
	state = deepcopy(state_0)
	for t in 1:max_t
		thetax = state.x - env.l*sin(state.θ)
		thetay = env.l*cos(state.θ)
		push!(xpositions,[state.x,thetax])
		push!(ypositions,[0,thetay])
		push!(xposcar,[state.x])
		push!(thetas,state.θ)
		push!(ws,state.w)
		push!(vs,state.v)
		push!(us,state.u)
		if state.u == 1
			break
		end
		#ids_dstate,discretized_state = discretize_state(state,env)
		#id_dstate = build_index_b(ids_dstate,env)
		push!(values,int_value(state.θ,state.w,state.u,state.v,state.x))
		actions,policy = optimal_policy_half(state,int_value,params,env)
		#Choosing the action with highest prob (empowerement style)
		#idx = findmax(policy)[2] 
		#Choosing action randomly according to policy
		push!(entropies,entropy(policy))
		idx = rand(Categorical(policy))
		action = actions[idx]
		push!(a_s,action)
		#For discretized dynamics
		#_,state_p = transition_b(state,action,env) 
		#For real dynamics
		states_p,probs_p = prob_transition(state,action,params,env)
		idx_s_p = rand(Categorical(probs_p))
		state_p = states_p[idx_s_p]
		state = deepcopy(state_p)
	#end
	end
	xpositions,ypositions,xposcar,thetas,ws,us,vs,a_s,values,entropies
end

# ╔═╡ c210a8ba-8b22-42b6-8d87-1d80dfe625bd
maxt_anim = 200

# ╔═╡ 37e6726b-71a3-46bf-9f36-2c38a478fc3e
md"## Many values of η"

# ╔═╡ b33ddf78-0254-4d1c-b0e6-9698a02ae089
ip_eta = inverted_pendulum_borders(M = 1, m = 0.1,l = 1.,Δt = 0.02, sizeθ = 31, sizew = 31,sizev = 31, sizex = 31, max_θ = 0.62, max_w = 3, a_s = [-40,-10,0,10,40], max_x = 1.8, max_v = 6, nactions = 5, γ = 0.99)

# ╔═╡ 48728be0-acbf-49d5-9ee9-f07fa562f199
#h_value_half,_,stop_h_half = h_iteration_half(params,ip_b,tolerance = 1E-1,n_iter = 100,verbose = true)
h_value_half = readdlm("values_half/h_value_beta_$(params.β)_g_$(ip_eta.γ)_eta_$(η_test).dat")

# ╔═╡ c5f05707-18e2-40ec-bce2-0da371914426
h_value_int_half = interpolate_value(h_value_half,ip_eta);

# ╔═╡ 09bf257f-fa33-4bb7-a2b9-77904cf528fe
begin
	time_histograms_half = time_histograms #10000
	xs_half, ys_half, xposcar_half, thetas_ep_half, ws_ep_half, us_ep_half, vs_ep_half, actions_half, values_half, entropies_half = create_episode_half(state_0_comp,h_value_int_half,time_histograms_half, params,ip_eta)
	#Check if it survived the whole episode
	length(xposcar_half)
	x_half = Any[]
	for i in 1:length(xposcar_half)
	 push!(x_half,xposcar_half[i][1])
	end
	length(x_half)
end

# ╔═╡ e4be43f1-28fb-4506-8b91-ea0f4c6d6304
begin
	p1half = plot(ylim = (-30,30),xlim=(-2,2))
	plot!(p1half,x_half,thetas_ep_half.*180/pi,bins = (100,100),st = :histogram2d,normed = :probability,clim = (0,cbarlim),cbar = true)
	plot(p1half,size = (400,300),margin = 6Plots.mm,grid = false)
	#savefig("stochastic_arena/histogram_beta_$(params.β).png")
end

# ╔═╡ 6b3f49a2-a0f5-4005-8034-ce8cb8d00d13
begin
	ps_trajectory_half = plot(1,xlabel = "Position",ylabel = labely,xlim = (-2.,2.),ylim=(-limy,limy),label = false, title = "H agent, \$\\beta = $(params.β)\$",marker = (:black,2))
	@gif for i in 1:t_trajectories#length(x_b)
		if angle
			push!(ps_trajectory_half,x_half[i],thetas_ep_half[i]*180/pi)
		else
			push!(ps_trajectory_half,x_half[i],vs_ep_half[i])
		end
	end every 10
end

# ╔═╡ 0fa694bf-d13f-4d62-8283-54accad831af
length([a for a in x_half if a > 0])/time_histograms

# ╔═╡ 15667883-b1fd-422a-ae10-74a22293acb9
anim_half = animate_w_borders(xposcar_half[1:maxt_anim],xs_half[1:maxt_anim],ys_half[1:maxt_anim],ip_b.max_x,maxt_anim,values_half,entropies_half[1:maxt_anim],ip_b,false, title = "Stochastic arena, \$\\beta = $(params.β), \\eta = $(params.η), \\gamma = $(ip_eta.γ)\$")

# ╔═╡ f2f0b55b-1a8c-47b8-b4dc-8f2f11556e13
gif(anim_half, fps = Int(1/ip_b.Δt))

# ╔═╡ c67e1e15-712d-4183-a75c-5361ad1ea5e7
print("values_half/h_value_beta_$(params.β)_g_$(ip_eta.γ)_eta_$(η_test).dat")

# ╔═╡ e1903c79-da4a-4d4a-9140-b5bb5b49133c
ηs_totry = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# ╔═╡ a438af2e-ca31-4427-bc74-e84301d1f9cf
# begin
# 	#Calculate q values for several ηs
# 	ηs_to_compute_value = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] #ηs_totry #ϵs_totry #[0.25,0.3,0.35,0.4]
# 	for (i,η) in enumerate(ηs_to_compute_value)
# 		par_sa = pars(β = 1,η = η)
# 		par_a = pars(β = 0, η = η)
# 		#println("epsilon = ", ϵ)
# 		h_val_sa, _, _ = h_iteration_half(par_sa,ip_eta,tolerance = 1E-3,n_iter = 1000)
# 		h_val_a, _, _ = h_iteration_half(par_a,ip_eta,tolerance = 1E-3,n_iter = 1000)
# 		writedlm("values_half/h_value_sa_g_$(ip_eta.γ)_eta_$(η).dat",h_val_sa)
# 		writedlm("values_half/h_value_a_g_$(ip_eta.γ)_eta_$(η).dat",h_val_a)
# 	end
# end

# ╔═╡ d7e60d83-fa62-451f-99f9-126f8cd0e821
# begin
# 	#Calculate q values for several βs
# 	βs_to_compute_value = collect(1.5:0.5:4.0)
# 	ηs_to_compute_value = [0.1,0.5,0.9]#[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] #ηs_totry #ϵs_totry #[0.25,0.3,0.35,0.4]
# 	for β in βs_to_compute_value
# 		for η in ηs_to_compute_value
# 		par_s = pars(β = β,η = η)
# 		#println("epsilon = ", ϵ)
# 		h_val, _, _ = h_iteration_half(par_s,ip_eta,tolerance = 1E-2,n_iter = 1000)
# 		writedlm("values_half/h_value_beta_$(β)_g_$(ip_eta.γ)_eta_$(η).dat",h_val)
# 		end
# 	end
# end

# ╔═╡ 5942cdba-6f0d-4128-8ce3-57826fafad0a
begin
	max_time_half = 10000
	num_episodes_half = 1000
	ηs = [0.1,0.5,0.9] #ηs_totry# [0.0,0.001,0.01,0.05] 
	βs = [-1.0,-0.6,0.0,0.6,1.0,1.5,2.0,2.5,3.0,3.5,4.0] #collect(1.5:0.5:4.0)
	survival_pcts_β = zeros(length(ηs),length(βs),num_episodes_half)
	time_stochastic_β = zeros(length(ηs),length(βs),num_episodes_half)
	# survival_pcts_sa = zeros(length(ηs),length(βs),num_episodes_half)
	# survival_pcts_a = zeros(length(ηs),num_episodes_half)
	# time_stochastic_sa = zeros(length(ηs),num_episodes_half)
	# time_stochastic_a = zeros(length(ηs),num_episodes_half)
	# for (i,η) in enumerate(ηs)
	# 	for (j,β) in enumerate(βs)
	# 		par = pars(β = β,η = η)
	# 		#par_a = pars(β = 0, η = η)
	# 		h_val = readdlm("values_half/h_value_beta_$(β)_g_$(ip_eta.γ)_eta_$(η).dat")
	# 		#h_val_a = readdlm("values_half/h_value_a_g_$(ip_eta.γ)_eta_$(η).dat")
	# 		h_val_int = interpolate_value(h_val,ip_eta)
	# 		#h_val_int_a = interpolate_value(h_val_a,ip_b)
	# 		Threads.@threads for k in 1:num_episodes_half
	# 			#println("j = ", j)
	# 			state0 = State(θ = rand(interval),x = -ip_eta.max_x/2, v = rand(interval),w = rand(interval), u = 2)
	# 			_,_, xposcar_half, _,_,_,_,_,_,_ = create_episode_half(state0,h_val_int,max_time_half, par,ip_eta)
	# 			# _,_, xposcar_half_a, _,_,_,_,_,_,_ = create_episode_half(state0,h_val_int_a,max_time_half, par_a,ip_eta)
	# 			x_half = Any[]
	# 			#x_half_a = Any[]
	# 			for k in 1:length(xposcar_half)
	# 			 push!(x_half,xposcar_half[k][1])
	# 			end
	# 			# for k in 1:length(xposcar_half_a)
	# 			#  push!(x_half_a,xposcar_half_a[k][1])
	# 			# end
				
	# 			time_stochastic_β[i,j,k] = length([a for a in x_half if a > 0])/length(xposcar_half)
	# 			#time_stochastic_a[i,j] = length([a for a in x_half_a if a > 0])/length(xposcar_half_a)
	# 			#survival_timesq[j] = length(xposcar_q)
	# 			#if length(xposcar_q) == max_time
	# 			survival_pcts_β[i,j,k] = length(xposcar_half)
	# 			#survival_pcts_a[i,j] = length(xposcar_half_a)
	# 			#end
	# 		end
	# 	end
	# end
	# writedlm("stochastic_arena/time_stochastic_betas_$(βs)_g_$(ip_eta.γ)_eta_$(ηs)_time_$(max_time_half).dat",time_stochastic_β)
	# #writedlm("stochastic_arena/time_stochastic_a_g_$(ip_eta.γ)_eta_$(ηs)_time_$(max_time_half).dat",time_stochastic_a)
	# writedlm("stochastic_arena/survival_pcts_betas_$(βs)_g_$(ip_eta.γ)_eta_$(ηs)_time_$(max_time_half).dat",survival_pcts_β)
	# #writedlm("stochastic_arena/survival_pcts_a_g_$(ip_eta.γ)_eta_$(ηs)_time_$(max_time_half).dat",survival_pcts_a)

	#Otherwise read from file
	time_stochastic_β = readdlm("stochastic_arena/time_stochastic_betas_$(βs)_g_$(ip_eta.γ)_eta_$(ηs)_time_$(max_time_half).dat")
	survival_pcts_β = readdlm("stochastic_arena/survival_pcts_betas_$(βs)_g_$(ip_eta.γ)_eta_$(ηs)_time_$(max_time_half).dat")
	time_stochastic_β = reshape(time_stochastic_β,length(ηs),length(βs),num_episodes_half)
	survival_pcts_β = reshape(survival_pcts_β,length(ηs),length(βs),num_episodes_half)
end;

# ╔═╡ 81f415cd-aea3-4c84-aede-c0040f5ac28c
begin
	ηss = collect(0.0:0.1:1.0)
	time_stochastic_sa = readdlm("stochastic_arena/time_stochastic_sa_g_$(ip_eta.γ)_eta_$(ηss)_time_$(max_time_half).dat")
	time_stochastic_a = readdlm("stochastic_arena/time_stochastic_a_g_$(ip_eta.γ)_eta_$(ηss)_time_$(max_time_half).dat")
	survival_pcts_sa = readdlm("stochastic_arena/survival_pcts_sa_g_$(ip_eta.γ)_eta_$(ηss)_time_$(max_time_half).dat")
	survival_pcts_a = readdlm("stochastic_arena/survival_pcts_a_g_$(ip_eta.γ)_eta_$(ηss)_time_$(max_time_half).dat")
	
end;

# ╔═╡ 6ebaa094-22b0-4e09-aa7f-eb3492dbf4f6
begin
	surv_means_half_β = plot(xticks = collect(0.0:1.0:4.0),xlabel = "β",ylabel = "Survived time steps",yticks = [5000,10000])
	colors = ["#C0C0C0","#696969","#000000"]
	for (i,η) in enumerate(ηs)
		plot!(surv_means_half_β,βs[3:end],mean(survival_pcts_β[i,3:end,:],dims =2),label = "\$\\eta\$ = $(η)",linewidth = 2.5,yerror = std(survival_pcts_β[i,3:end,:],dims = 2)./sqrt(length(survival_pcts_β[i,1,:])),markerstrokewidth = 1,markersize = 3,color = colors[i])
		plot!(surv_means_half_β, βs[3:end], mean(survival_pcts_β[i,3:end,:],dims =2),label = false,color = colors[i],markershape = :circle,markersize = 2.5,st = :scatter)
	end
	plot(surv_means_half_β, grid = false, legend_position = :bottomright,margin = 2Plots.mm,size = (450,300),bg_legend = :transparent,fg_legend = :transparent,fgguide = :black)
	#savefig("stochastic_arena/h_betas_$(βs)_survival_time_$(max_time_half).pdf")
end

# ╔═╡ b28859d1-7c4d-438d-a490-c6407365cd6a
begin
	surv_means_half = plot(xticks = collect(0.0:0.2:1),xlabel = "η",ylabel = "Survived time steps",yticks = [5000,10000])
		plot!(surv_means_half, ηss, mean(survival_pcts_a,dims =2).*ones(length(ηss)),label = "\$\\beta = 0\$",linewidth = 2.5,yerror = std(survival_pcts_a,dims = 2)./sqrt(length(survival_pcts_a[1,:])),color = colors[1])
	plot!(surv_means_half, ηss, mean(survival_pcts_a,dims =2).*ones(length(ηss)),label = false,st = :scatter,markersize = 2.5,color = colors[1])
	plot!(surv_means_half, ηss, mean(survival_pcts_sa,dims =2).*ones(length(ηss)),label = "\$\\beta = 1\$",linewidth = 2.5,yerror = std(survival_pcts_sa,dims = 2)./sqrt(length(survival_pcts_sa[1,:])),color = colors[2])
	plot!(surv_means_half, ηss, mean(survival_pcts_sa,dims =2).*ones(length(ηss)),label = false,st = :scatter,markersize = 2,5,color = colors[2])

	

	plot(surv_means_half, grid = false, legend_position = :bottomleft,margin = 2Plots.mm,size = (450,300),bg_legend = :white,fg_legend = :white,fgguide = :black)
	#savefig("stochastic_arena/h_sa_vs_a_survival_time_$(max_time_half).pdf")
end

# ╔═╡ d7d217ce-2a0b-4e18-8aa8-8b5e3af08363
begin
	times_means_half_β = plot(ylim=(0,1),xticks = collect(0.0:1.0:4.0),yticks = [0.0,0.25,0.5,0.75,1.0])#,xlabel = "β",ylabel = "Time fraction right")
	for (i,η) in enumerate(ηs)
		plot!(times_means_half_β,βs[3:end],mean(time_stochastic_β[i,3:end,:],dims =2),label = "\$\\eta\$ = $(η)",linewidth = 2.5,yerror = std(time_stochastic_β[i,3:end,:],dims = 2)./sqrt(length(time_stochastic_β[i,3,:])),markerstrokewidth = 1,markersize = 3,color = colors[i])
		plot!(times_means_half_β, βs[3:end], mean(time_stochastic_β[i,3:end,:],dims =2),label = false,linewidth = 2,color = colors[i],markershape = :circle,markersize = 2.5,st = :scatter)
		#plot!(times_means_half_β, βs, mean(time_stochastic_β[i,:,:],dims =2),label = "η = $(η)",linewidth = 2.5,yerror = std(time_stochastic_β[i,:,:],dims = 2)./sqrt(length(time_stochastic_β[i,1,:])))
	end


	plot(times_means_half_β, grid = false, legend_position = :topright,margin = 2Plots.mm,size = (450,300),bg_legend = :transparent,fg_legend = :transparent,fgguide = :black)
	#savefig("stochastic_arena/h_betas_$(βs)_timeonpositive.pdf")
end

# ╔═╡ b96306e1-2c38-4245-8d0a-87d503ab9df8
begin
	times_means_half = plot(ylim=(0,1),yticks = [0.0,0.25,0.5,0.75,1.0],xticks = collect(0.0:0.2:1))#xlabel = "η",ylabel = "Fraction of time on \$x > 0\$")
	plot!(times_means_half, ηss, mean(time_stochastic_a,dims =2).*ones(length(ηss)),label = "\$\\beta = 0\$",linewidth = 2.5,yerror = std(time_stochastic_a,dims = 2)./sqrt(length(time_stochastic_a[1,:])),color = "#C0C0C0",markercolor = "#C0C0C0")
	plot!(times_means_half, ηss, mean(time_stochastic_a,dims =2).*ones(length(ηss)),label = false,markershape = :circle,markersize = 2.5,color = "#C0C0C0",st = :scatter)
	plot!(times_means_half, ηss, mean(time_stochastic_sa,dims =2).*ones(length(ηss)),label = "\$\\beta = 1\$",linewidth = 2.5,yerror = std(time_stochastic_sa,dims = 2)./sqrt(length(time_stochastic_sa[1,:])),color = "#696969")
	plot!(times_means_half, ηss, mean(time_stochastic_sa,dims =2).*ones(length(ηss)),label = false,linewidth = 2.5,color = "#696969",markershape = :circle,markersize = 2.5,st = :scatter)

	plot(times_means_half, grid = false, legend_position = :topright,margin = 2Plots.mm,size = (450,300),bg_legend = :transparent,fg_legend = :transparent,fgguide = :black)
	#savefig("stochastic_arena/h_sa_vs_a_timeonpositive.pdf")
end

# ╔═╡ 930f7d64-29ef-4e7b-826d-66243e9724e9
md"# Death analysis (not part of paper)"

# ╔═╡ 03fc1bf9-bcf5-44e9-9542-da9325639907
border_sizes = [2.8,2.4,2.0,1.6,1.2,0.8,0.4,0.2] #[2.4,2.0,1.6,1.2,0.8]

# ╔═╡ bcc12d64-1eb8-4edb-b042-57e70ce3b641
begin
	#For R agent
	γs_R = [0.82,0.84,0.86,0.88,0.9,0.92]
	#For H agent 
	γs_H = [0.9,0.91,0.92,0.94,0.96,0.97]
	#γs = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97]
end

# ╔═╡ d9df62ae-47e7-4bde-907e-9eff4251c17f
ϵ_border = 0.4

# ╔═╡ 20061a57-ac97-409d-8a88-7decef927609
# begin
# 	b_par = 2.4
# 	#γ = 0.92
# 	#Tolerance for convergence of value iteration
# 	tol_borders = 1E-3
# 	#Array of parameters
# 	parameter = [0.84,0.82] #γs #border_sizes
# 	#Calculate q values for several borders
# 	for (i,p) in enumerate(parameter)
# 		γ = p
# 		ip_border = inverted_pendulum_borders(M = 1, m = 0.1,l = 1.,Δt = 0.02, sizeθ = 31, sizew = 31,sizev = 31, sizex = 31, max_θ = 0.62, max_w = 3, a_s = [-50,-10,0,10,50], max_x = b_par, max_v = 3, nactions = 5, γ = γ)
# 		ϵ = ϵ_border
# 		#h_val,_,_= iteration(ip_border,tol_borders,1200,δ = δ_r)
# 		q_val, _, _ = Q_iteration(ip_border,ϵ,tol_borders,1200,δ = δ_r)
# 		#writedlm("values_gamma/h_value_g_$(ip_border.γ)_nstates_$(ip_border.nstates)_xlim_$(b_par).dat",h_val)
# 		writedlm("values_gamma/q_value_g_$(ip_border.γ)_eps_$(ϵ)_nstates_$(ip_border.nstates)_xlim_$(b_par).dat",q_val)
# 	end
# end

# ╔═╡ 4ba1e06b-f1b9-4faa-99e8-abd133a9052b
begin
	max_time_gammas = 1000
	num_episodes_gammas = 10000
	n_blocks_gammas = 50
	#To compute the survival times for various environments, it takes a long time
	survival_R_gammas = zeros(length(γs_R),num_episodes_gammas)
	survival_H_gammas = zeros(length(γs_R),num_episodes_gammas)
	#Per agent entropy
	# entropies_R_borders = zeros(length(border_sizes),num_episodes_borders)
	# entropies_H_borders = zeros(length(border_sizes),num_episodes_borders)
	#Many agent entropy
	entropies_R_gammas = zeros(length(γs_R),n_blocks_gammas)
	entropies_H_gammas = zeros(length(γs_R),n_blocks_gammas)
	b_gamma = 2.4
	# for (i,γ) in enumerate(γs_R)
	# 	ip_r = inverted_pendulum_borders(M = 1, m = 0.1,l = 1.,Δt = 0.02, sizeθ = 31, sizew = 31,sizev = 31, sizex = 31, max_θ = 0.62, max_w = 3, a_s = [-50,-10,0,10,50], max_x = b_gamma, max_v = 3, nactions = 5, γ = γ)
	# 	ip_h = inverted_pendulum_borders(M = 1, m = 0.1,l = 1.,Δt = 0.02, sizeθ = 31, sizew = 31,sizev = 31, sizex = 31, max_θ = 0.62, max_w = 3, a_s = [-50,-10,0,10,50], max_x = b_gamma, max_v = 3, nactions = 5, γ = γs_H[i])
	# 	bins = (ip_r.θs,ip_r.ws,ip_r.vs,ip_r.xs)
	# 	q_val = readdlm("values_gamma/q_value_g_$(ip_r.γ)_eps_$(ϵ_border)_nstates_$(ip_r.nstates)_xlim_$(b_gamma).dat")
	# 	h_val = readdlm("values_gamma/h_value_g_$(ip_h.γ)_nstates_$(ip_h.nstates)_xlim_$(b_gamma).dat")
	# 	q_val_int = interpolate_value(q_val,ip_r)
	# 	h_val_int = interpolate_value(h_val,ip_h)
	# 	thetas_h = Any[]
	# 	thetas_r = Any[]
	# 	ws_h = Any[]
	# 	ws_r = Any[]
	# 	vs_h = Any[]
	# 	vs_r = Any[]
	# 	xs_h = Any[]
	# 	xs_r = Any[]
	# 	block = 1
	# 	for j in 1:num_episodes_gammas
	# 		#println("j = ", j)
	# 		state0 = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval), u = 2)
	# 		xs_q, ys_q, xposcar_q, thetas_ep_q, ws_ep_q, us_ep_q, vs_ep_q, actions_q = create_episode_q(state0, q_val_int, ϵ_border, max_time_gammas, ip_r)
	# 		xs_b, ys_b, xposcar_b, thetas_ep_b, ws_ep_b, us_ep_b, vs_ep_b, actions_b = create_episode_b(state0,h_val_int,max_time_gammas, ip_h)
	# 		#For many agent entropy
	# 		survival_R_gammas[i,j] = length(xposcar_q)
	# 		survival_H_gammas[i,j] = length(xposcar_b)
	# 		push!(thetas_h,thetas_ep_b...)
	# 		push!(thetas_r,thetas_ep_q...)
	# 		push!(ws_h,ws_ep_b...)
	# 		push!(ws_r,ws_ep_q...)
	# 		push!(vs_h,vs_ep_b...)
	# 		push!(vs_r,vs_ep_q...)
	# 		push!(xs_h,[xposcar_b[i][1] for i in 1:(length(xposcar_b))]...)
	# 		push!(xs_r,[xposcar_q[i][1] for i in 1:(length(xposcar_q))]...)
	# 		if j >= block*num_episodes_gammas/n_blocks_gammas
	# 			h_r = fit(Histogram, (thetas_r,ws_r, vs_r,xs_r), bins)
	# 			h_h = fit(Histogram, (thetas_h,ws_h, vs_h,xs_h), bins)
	# 			h_r_n = normalize(h_r,mode =:probability)
	# 			h_h_n = normalize(h_h,mode =:probability)
	# 			entropies_R_gammas[i,block] = entropy(h_r_n.weights)
	# 			entropies_H_gammas[i,block] = entropy(h_h_n.weights)
	# 			thetas_h = Any[]
	# 			thetas_r = Any[]
	# 			ws_h = Any[]
	# 			ws_r = Any[]
	# 			vs_h = Any[]
	# 			vs_r = Any[]
	# 			xs_h = Any[]
	# 			xs_r = Any[]
	# 			block += 1 
	# 		end
	# 		# For per agent entropy
	# 		# bins = (ip_border.θs,ip_border.ws,ip_border.vs,ip_border.xs)
	# 		# h_r = fit(Histogram, (thetas_ep_q,ws_ep_q, vs_ep_q,[xposcar_q[i][1] for i in 1:(length(xposcar_q))]), bins)
	# 		# h_h = fit(Histogram, (thetas_ep_b,ws_ep_b, vs_ep_b,[xposcar_b[i][1] for i in 1:(length(xposcar_b))]), bins)
	# 		# h_r_n = normalize(h_r,mode =:probability)
	# 		# h_h_n = normalize(h_h,mode =:probability)
	# 		# entropies_R_borders[i,j] = entropy(h_r_n.weights)
	# 		# entropies_H_borders[i,j] = entropy(h_h_n.weights)
	# 		#end
	# 		end
	# 	#h_r = fit(Histogram, (thetas_r,ws_r, vs_r,xs_r), bins)
	# 	#h_h = fit(Histogram, (thetas_h,ws_h, vs_h,xs_h), bins)
	# 	#h_r_n = normalize(h_r,mode =:probability)
	# 	#h_h_n = normalize(h_h,mode =:probability)
	# 	#entropies_R_borders[i] = entropy(h_r_n.weights)
	# 	#entropies_H_borders[i] = entropy(h_h_n.weights)
	# end
	
	#Read from file
	survival_R_gammas = readdlm("survivals/survival_gammas_R_epsilon_$(ϵ_border)_maxtime_$(max_time_gammas).dat")
	survival_H_gammas = readdlm("survivals/survival_gammas_H_maxtime_$(max_time_gammas).dat")
	entropies_H_gammas = readdlm("survivals/entropy_gammas_H_maxtime_$(max_time_gammas).dat")
	entropies_R_gammas = readdlm("survivals/entropy_gammas_R_epsilon_$(ϵ_border)_maxtime_$(max_time_gammas).dat")
end

# ╔═╡ 92356131-5c26-4257-a148-2b9f49f2c9c6
begin
	plot(legend_position = :right,xlabel = "Mean lifetime",ylabel = "Many agent state entropy")
	plot!(mean(survival_R_gammas,dims = 2),xerror = std(survival_R_gammas,dims = 2)/sqrt(length(survival_R_gammas[1,:])),mean(entropies_R_gammas,dims = 2),yerror = std(entropies_R_gammas,dims = 2)/sqrt(length(entropies_R_gammas[1,:])),label = "R agent")
	
	plot!(mean(survival_H_gammas,dims = 2),xerror = std(survival_H_gammas,dims = 2)/sqrt(length(survival_H_gammas[1,:])),mean(entropies_H_gammas,dims = 2),yerror = std(entropies_H_gammas,dims = 2)/sqrt(length(entropies_H_gammas[1,:])),label = "H agent")
	#savefig("many_agent_entropy_gamma2.pdf")
end

# ╔═╡ cae8d738-94f5-47ec-a7cf-4da51c069c75
# begin
# 	writedlm("survivals/survival_gammas_R_epsilon_$(ϵ_border)_maxtime_$(max_time_gammas).dat",survival_R_gammas)
# 	writedlm("survivals/survival_gammas_H_maxtime_$(max_time_gammas).dat",survival_H_gammas)
# 	writedlm("survivals/entropy_gammas_R_epsilon_$(ϵ_border)_maxtime_$(max_time_gammas).dat",entropies_R_gammas)
# 	writedlm("survivals/entropy_gammas_H_maxtime_$(max_time_gammas).dat",entropies_H_gammas)
# end

# ╔═╡ ac58190d-05bd-4e3e-9797-9bc1ee32b545
#writedlm("survivals/survival_borders_R_epsilon_$(ϵ_border)_maxtime_$(max_time_borders).dat",survival_R_borders)
# writedlm("survivals/survival_borders_H_maxtime_$(max_time_borders).dat",survival_H_borders)
#writedlm("survivals/entropy_borders_R_epsilon_$(ϵ_border)_maxtime_$(max_time_borders).dat",entropies_R_borders)
# writedlm("survivals/entropy_borders_H_maxtime_$(max_time_borders).dat",entropies_H_borders)

# ╔═╡ 15928993-d9fc-4934-a36a-90fb54a322ba
begin
	max_time_borders = 1000
	num_episodes_borders = 50000
	n_blocks = 50
	# #To compute the survival times for various environments, it takes a long time
	# survival_R_borders = zeros(length(border_sizes),num_episodes_borders)
	# survival_H_borders = zeros(length(border_sizes),num_episodes_borders)
	# #Per agent entropy
	# # entropies_R_borders = zeros(length(border_sizes),num_episodes_borders)
	# # entropies_H_borders = zeros(length(border_sizes),num_episodes_borders)
	# #Many agent entropy
	# entropies_R_borders = zeros(length(border_sizes),n_blocks)
	# entropies_H_borders = zeros(length(border_sizes),n_blocks)
	# for (i,b) in enumerate(border_sizes)
	# 	ip_border = inverted_pendulum_borders(M = 1, m = 0.1,l = 1.,Δt = 0.02, sizeθ = 31, sizew = 31,sizev = 31, sizex = 31, max_θ = 0.62, max_w = 3, a_s = [-50,-10,0,10,50], max_x = b, max_v = 3, nactions = 5, γ = 0.92)
	# 	bins = (ip_border.θs,ip_border.ws,ip_border.vs,ip_border.xs)
	# 	q_val = readdlm("values_borders/q_value_g_$(ip_border.γ)_eps_$(ϵ_border)_nstates_$(ip_border.nstates)_xlim_$(b).dat")
	# 	h_val = readdlm("values_borders/h_value_g_$(ip_border.γ)_nstates_$(ip_border.nstates)_xlim_$(b).dat")
	# 	q_val_int = interpolate_value(q_val,ip_border)
	# 	h_val_int = interpolate_value(h_val,ip_border)
	# 	thetas_h = Any[]
	# 	thetas_r = Any[]
	# 	ws_h = Any[]
	# 	ws_r = Any[]
	# 	vs_h = Any[]
	# 	vs_r = Any[]
	# 	xs_h = Any[]
	# 	xs_r = Any[]
	# 	block = 1
	# 	for j in 1:num_episodes_borders
	# 		#println("j = ", j)
	# 		state0 = State(θ = rand(interval),x = rand(interval), v = rand(interval),w = rand(interval), u = 2)
	# 		xs_q, ys_q, xposcar_q, thetas_ep_q, ws_ep_q, us_ep_q, vs_ep_q, actions_q = create_episode_q(state0, q_val_int, ϵ_border, max_time_borders, ip_border)
	# 		xs_b, ys_b, xposcar_b, thetas_ep_b, ws_ep_b, us_ep_b, vs_ep_b, actions_b = create_episode_b(state0,h_val_int,max_time_borders, ip_border)
	# 		#For many agent entropy
	# 		survival_R_borders[i,j] = length(xposcar_q)
	# 		survival_H_borders[i,j] = length(xposcar_b)
	# 		push!(thetas_h,thetas_ep_b...)
	# 		push!(thetas_r,thetas_ep_q...)
	# 		push!(ws_h,ws_ep_b...)
	# 		push!(ws_r,ws_ep_q...)
	# 		push!(vs_h,vs_ep_b...)
	# 		push!(vs_r,vs_ep_q...)
	# 		push!(xs_h,[xposcar_b[i][1] for i in 1:(length(xposcar_b))]...)
	# 		push!(xs_r,[xposcar_q[i][1] for i in 1:(length(xposcar_q))]...)
	# 		if j >= block*num_episodes_borders/n_blocks
	# 			h_r = fit(Histogram, (thetas_r,ws_r, vs_r,xs_r), bins)
	# 			h_h = fit(Histogram, (thetas_h,ws_h, vs_h,xs_h), bins)
	# 			h_r_n = normalize(h_r,mode =:probability)
	# 			h_h_n = normalize(h_h,mode =:probability)
	# 			entropies_R_borders[i,block] = entropy(h_r_n.weights)
	# 			entropies_H_borders[i,block] = entropy(h_h_n.weights)
	# 			thetas_h = Any[]
	# 			thetas_r = Any[]
	# 			ws_h = Any[]
	# 			ws_r = Any[]
	# 			vs_h = Any[]
	# 			vs_r = Any[]
	# 			xs_h = Any[]
	# 			xs_r = Any[]
	# 			block += 1 
	# 		end
	# 		# For per agent entropy
	# 		# bins = (ip_border.θs,ip_border.ws,ip_border.vs,ip_border.xs)
	# 		# h_r = fit(Histogram, (thetas_ep_q,ws_ep_q, vs_ep_q,[xposcar_q[i][1] for i in 1:(length(xposcar_q))]), bins)
	# 		# h_h = fit(Histogram, (thetas_ep_b,ws_ep_b, vs_ep_b,[xposcar_b[i][1] for i in 1:(length(xposcar_b))]), bins)
	# 		# h_r_n = normalize(h_r,mode =:probability)
	# 		# h_h_n = normalize(h_h,mode =:probability)
	# 		# entropies_R_borders[i,j] = entropy(h_r_n.weights)
	# 		# entropies_H_borders[i,j] = entropy(h_h_n.weights)
	# 		#end
	# 		end
	# 	#h_r = fit(Histogram, (thetas_r,ws_r, vs_r,xs_r), bins)
	# 	#h_h = fit(Histogram, (thetas_h,ws_h, vs_h,xs_h), bins)
	# 	#h_r_n = normalize(h_r,mode =:probability)
	# 	#h_h_n = normalize(h_h,mode =:probability)
	# 	#entropies_R_borders[i] = entropy(h_r_n.weights)
	# 	#entropies_H_borders[i] = entropy(h_h_n.weights)
	# end
	#writedlm("survivals/survival_borders_R_epsilon_$(ϵ_border)_maxtime_$(max_time_borders).dat",survival_R_borders)
	# writedlm("survivals/survival_borders_H_maxtime_$(max_time_borders).dat",survival_H_borders)
	#writedlm("survivals/entropy_borders_R_epsilon_$(ϵ_border)_maxtime_$(max_time_borders).dat",entropies_R_borders)
	# writedlm("survivals/entropy_borders_H_maxtime_$(max_time_borders).dat",entropies_H_borders)
	
	#Read from file
	survival_R_borders = readdlm("survivals/survival_borders_R_epsilon_$(ϵ_border)_maxtime_$(max_time_borders).dat")
	survival_H_borders = readdlm("survivals/survival_borders_H_maxtime_$(max_time_borders).dat")
	entropies_H_borders = readdlm("survivals/entropy_borders_H_maxtime_$(max_time_borders).dat")
	entropies_R_borders = readdlm("survivals/entropy_borders_R_epsilon_$(ϵ_border)_maxtime_$(max_time_borders).dat")
end

# ╔═╡ a07bb87e-8321-4aa8-9555-ae33a51c5fa7
# begin
# 	writedlm("survivals/survival_borders_g_0.92_R_epsilon_$(ϵ_border)_maxtime_$(max_time_borders).dat",survival_R_borders)
# 	writedlm("survivals/survival_borders_g_0.92_H_maxtime_$(max_time_borders).dat",survival_H_borders)
# 	writedlm("survivals/entropy_borders_g_0.92_R_epsilon_$(ϵ_border)_maxtime_$(max_time_borders).dat",entropies_R_borders)
# 	writedlm("survivals/entropy_borders_g_0.92_H_maxtime_$(max_time_borders).dat",entropies_H_borders)
# end

# ╔═╡ cbb6c21f-81fe-4859-be21-2ff3759321fb
begin
	plot(legend_position = :right,xlabel = "Mean lifetime",ylabel = "Many agent state entropy")
	plot!(mean(survival_R_borders,dims = 2),xerror = std(survival_R_borders,dims = 2)/sqrt(length(survival_R_borders[1,:])),mean(entropies_R_borders,dims = 2),yerror = std(entropies_R_borders,dims = 2)/sqrt(length(entropies_R_borders[1,:])),label = "R agent")
	
	plot!(mean(survival_H_borders,dims = 2),xerror = std(survival_H_borders,dims = 2)/sqrt(length(survival_H_borders[1,:])),mean(entropies_H_borders,dims = 2),yerror = std(entropies_H_borders,dims = 2)/sqrt(length(entropies_H_borders[1,:])),label = "H agent")
	#savefig("many_agent_entropy_borders.pdf")
end

# ╔═╡ cb24610c-2eac-4054-a17c-0ef4f3668f6c
mean(survival_R_borders,dims = 2),mean(entropies_R_borders,dims = 2)

# ╔═╡ 5721178c-e93e-4f2e-b39b-58ebdf9edfb8
entropies_R_borders

# ╔═╡ 5800b6d5-8593-4f1d-95f6-9e3353f63eb2
entropies_H_borders

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
ZipFile = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"

[compat]
Distributions = "~0.25.75"
Interpolations = "~0.14.6"
Parameters = "~0.12.3"
Plots = "~1.35.3"
PlutoUI = "~0.7.43"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.4"
ZipFile = "~0.10.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "bb215c73e0143aeb6f63b8e09ffa2edd888d0a34"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "84259bb6172806304b9101094a7cc4bc6f56dbc6"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "64df3da1d2a26f4de23871cd1b6482bb68092bd5"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.3"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "3ca828fe1b75fa84b021a7860bd039eaea84d2f2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.3.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "46d2680e618f8abd007bce0c3026cb0c4a8f2032"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.12.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0d7d213133d948c56e8c2d9f4eab0293491d8e4a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.75"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "87519eb762f85534445f5cda35be12e32759ee14"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.4"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "cf7bf90e483228f6c988e474b420064e5351b892"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.69.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fb83fbe02fe57f2c068013aa94bcdf6760d3a7a7"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "4abede886fcba15cd5fd041fef776b230d004cee"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.4.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "842dd89a6cb75e02e85fdd75c760cdc43f5d6863"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.6"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "41d162ae9c868218b1f3fe78cba878aa348c2d26"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.1.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "6872f9594ff273da6d13c7c1a1545d5a8c7d0c1c"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.6"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "efe9c8ecab7a6311d4b91568bd6c88897822fabe"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "440165bf08bc500b8fe4a7be2dc83271a00c0716"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "5a9ea4b9430d511980c01e9f7173739595bbd335"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.2"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "ebe81469e9d7b471d7ddb611d9e147ea16de0add"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.2.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "595c0b811cf2bab8b0849a70d9bd6379cc1cfb52"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "524d9ff1b2f4473fef59678c06f9f77160a204b1"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.35.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "2777a5c2c91b3145f5aa75b61bb4c2eb38797136"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.43"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "3c009334f45dfd546a16a57960a821a1a023d241"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.5.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "612a4d76ad98e9722c8ba387614539155a59e30c"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.0"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "9b1c0c8e9188950e66fc28f40bfe0f8aac311fe0"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.7"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "efd23b378ea5f2db53a55ae53d3133de4e080aa9"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.16"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "f86b3a049e5d05227b10e15dbb315c5b90f14988"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.9"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e0d5bc26226ab1b7648278169858adcfbd861780"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "2d7164f7b8a066bcfa6224e67736ce0eb54aef5b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.9.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "ef4f23ffde3ee95114b461dc667ea4e6906874b2"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╠═63a9f3aa-31a8-11ec-3238-4f818ccf6b6c
# ╠═1525f483-b6ac-4b12-90e6-31e97637d282
# ╠═11f37e01-7958-4e79-a8bb-06b92d7cb9ed
# ╠═0a29d50e-df0a-4f54-b588-faa6aee2e983
# ╠═18382e9c-e812-49ff-84cc-faad709bc4c3
# ╠═379318a2-ea2a-4ac1-9046-0fdfe8c102d4
# ╟─40ee8d98-cc45-401e-bd9d-f9002bc4beba
# ╠═4ec0dd6d-9d3d-4d30-8a19-bc91600d9ec2
# ╟─f997fa16-69e9-4df8-bed9-7067e1a5537d
# ╟─2f31b286-11aa-443e-a3dc-c021e6fc276c
# ╟─b1d3fff1-d980-4cc8-99c3-d3db7a71bf60
# ╟─9396a0d1-6036-44ec-98b7-16df4d150b54
# ╟─cfdf3a8e-a377-43ef-8a76-63cf78ce6a16
# ╟─ffe32878-8732-46d5-b57c-9e9bb8e6dd74
# ╟─784178c3-4afc-4c65-93e1-4265e1faf817
# ╟─fa56d939-09d5-4b63-965a-38016c957fbb
# ╟─05b8a93c-cefe-473a-9f8e-3e82de0861b2
# ╟─f92e8c6d-bf65-4786-8782-f38847a7fb7a
# ╟─f9121267-8d7e-40b1-b9cc-c8da3f45cdb8
# ╟─36601cad-1ba9-48f2-8463-a58f98bedd34
# ╟─ce8bf897-f240-44d8-ae39-cf24eb115704
# ╟─b975eaf8-6a94-4e39-983f-b0fb58dd70a1
# ╟─e0fdce26-41b9-448a-a86a-6b29b68a6782
# ╠═bbe19356-e00b-4d90-b704-db33e0b75743
# ╟─b38cdb2b-f570-41f4-8996-c7e41551f374
# ╠═dafbc66b-cbc2-41c9-9a42-da5961d2eaa6
# ╠═2735cbf5-790d-40b4-ac32-a413bc1d530a
# ╟─79372251-157d-43a0-9560-4727fbd36ea9
# ╠═2f1b6992-3082-412d-b966-2b4b278b2ed0
# ╠═aa4229d3-e221-4a87-8a18-ed376d33d3ac
# ╟─10a47fa8-f235-4455-892f-9b1457b1f82c
# ╠═a54788b9-4b1e-4066-b963-d04008bcc242
# ╠═69128c7a-ddcb-4535-98af-24fc91ec0b7e
# ╟─e099528b-37a4-41a2-836b-69cb3ceda2f5
# ╠═6b9b1c38-0ed2-4fe3-9326-4670a33e7765
# ╠═7c04e2c3-4157-446d-a065-4bfe7d1931fd
# ╠═a0b85a14-67af-42d6-b231-1c7d0c293f6e
# ╠═8a59e209-9bb6-4066-b6ca-70dac7da33c3
# ╟─08ecfbed-1a7c-43d4-ade7-bf644eeb6eee
# ╟─21472cb5-d968-4306-b65b-1b25f522dd4d
# ╠═c5c9bc66-f554-4fa8-a9f3-896875a50627
# ╠═370a2da6-6de0-44c0-9f70-4e676769f59b
# ╠═288aa06b-e07b-41cc-a51f-49f780c634b8
# ╠═9230de54-3ee3-4242-bc34-25a38edfbb6b
# ╟─24a4ba06-d3ae-4c4b-9ab3-3852273c2fd4
# ╟─8b38bccf-a762-439e-b19b-65e803d3c8f6
# ╟─63fe2f55-f5ce-4022-99ee-3bd4e14e6352
# ╠═14314bcc-661a-4e84-886d-20c89c07a28e
# ╠═c087392c-c411-4368-afcc-f9a104856884
# ╟─6107a0ce-6f01-4d0b-bd43-78f2125ac185
# ╟─e181540d-4334-47c4-b35d-99023c89a2c8
# ╟─31a1cdc7-2491-42c1-9988-63650dfaa3e3
# ╟─c05d12f3-8b8a-4f34-bebc-77e376a980d0
# ╠═87c0c3cb-7059-42ae-aed8-98a0ef2eb55f
# ╟─f9b03b4d-c521-4456-b0b9-d4a301d8a813
# ╠═355db008-9661-4e54-acd5-7c2c9ba3c7f5
# ╠═8a6b67d0-8008-4a4c-be7c-0c0b76311385
# ╠═c2105bee-c29d-4853-9388-31c405283395
# ╠═564cbc7a-3125-4b67-843c-f4c74ccef51f
# ╠═d20c1afe-6d5b-49bf-a0f2-a1bbb21c709f
# ╠═e9687e3f-be56-44eb-af4d-f169558de0fd
# ╟─e9a01524-90b1-4249-a51c-ec8d1624be5b
# ╟─4893bf14-446c-46a7-b695-53bac123ff99
# ╠═7edf8ddf-2b6e-4a4b-8181-6b8bbdd22841
# ╠═c98025f8-f942-498b-9368-5d524b141c62
# ╠═d8642b83-e824-429e-ac3e-70e875a47d1a
# ╟─90aff8bb-ed69-40d3-a22f-112f713f4b93
# ╟─85e789c6-8322-4264-ab70-dc33d64c4de4
# ╠═43ce27d1-3246-4045-b3aa-a99bcf25cbaa
# ╠═0832d5e6-7e92-430f-afe3-ddb5e55dc591
# ╟─4fad0692-05dc-4c3b-9aae-cd9a43519e51
# ╠═973b56e5-ef3c-4328-ad26-3ab63650537e
# ╠═3d567640-78d6-4f76-b13e-95be6d5e9c64
# ╠═06f064cf-fc0d-4f65-bd6b-ddb6e4154f6c
# ╠═ac43808a-ef2a-472d-9a9e-c77257aaa535
# ╠═da37a5c8-b8ef-4131-a153-50c21461d9c4
# ╠═a8b1a61c-6bd0-417b-8d58-8c8f8d77da7e
# ╟─1a6c956e-38cc-4217-aff3-f303af0282a4
# ╠═e169a987-849f-4cc2-96bc-39f234742d93
# ╠═ae15e079-231e-4ea2-aabb-ee7d44266c6d
# ╟─a4b26f44-319d-4b90-8fee-a3ab2418dc47
# ╠═94da3cc0-6763-40b9-8773-f2a1a2cbe507
# ╠═69b6d1f1-c46c-43fa-a21d-b1f61fcd7c55
# ╠═7a27d480-a9cb-4c26-91cd-bf519e8b35fa
# ╠═2c4375df-5063-43ea-9578-69ec94834362
# ╠═339a47a1-2b8a-4dc2-9adc-530c53d66fb1
# ╠═8c819d68-3981-4073-b58f-8fde5b73be33
# ╠═b367ccc6-934f-4b18-b1db-05286111958f
# ╠═096a58e3-f417-446e-83f0-84a333880680
# ╠═c95dc4f2-1f54-4266-bf23-7d24cee7b3d4
# ╠═d2139819-83a1-4e2b-a605-91d1e686b9a3
# ╠═fa9ae665-465d-4f3e-b8cd-002c80420adb
# ╠═6cc17343-c725-4653-ad48-b3535a53b09e
# ╠═9fa87325-4af3-4ec0-a716-80652dcf2ace
# ╠═1aace394-c7da-421a-bd4c-e7c7d8b36231
# ╠═c9e4bc33-046f-4ebd-9da7-a768886107ef
# ╠═f4b1a0bf-aff3-4b49-b15f-e5fdf31d969c
# ╠═bde8e418-9391-4c5a-a754-5ae1e3215e66
# ╠═ac2816ee-c066-4063-ae74-0a143df37a9c
# ╠═d0c487cb-041f-4c8d-9054-e5f3cfad1ed4
# ╟─e25f575e-91d5-479e-a752-a831a0692f26
# ╠═0e7f28f3-53b2-431d-afd9-d2fe6c511863
# ╠═e0c1c42b-4327-4ad8-b097-92bf08912e3e
# ╠═47c63b41-4385-479f-b0e9-afae0ed08058
# ╟─d3cbbcca-b43a-421e-b29f-4388a409de41
# ╟─bcff0238-4182-4407-a8b7-f19e6b700906
# ╟─581515a9-3d39-44ac-be92-e3049a36a15d
# ╟─5a71d63b-a63e-4ad7-abc4-36c2f2c61711
# ╠═c5f05707-18e2-40ec-bce2-0da371914426
# ╠═09bf257f-fa33-4bb7-a2b9-77904cf528fe
# ╠═e4be43f1-28fb-4506-8b91-ea0f4c6d6304
# ╠═6b3f49a2-a0f5-4005-8034-ce8cb8d00d13
# ╠═0fa694bf-d13f-4d62-8283-54accad831af
# ╠═c210a8ba-8b22-42b6-8d87-1d80dfe625bd
# ╠═15667883-b1fd-422a-ae10-74a22293acb9
# ╠═f2f0b55b-1a8c-47b8-b4dc-8f2f11556e13
# ╠═48728be0-acbf-49d5-9ee9-f07fa562f199
# ╠═c67e1e15-712d-4183-a75c-5361ad1ea5e7
# ╟─37e6726b-71a3-46bf-9f36-2c38a478fc3e
# ╠═b33ddf78-0254-4d1c-b0e6-9698a02ae089
# ╠═e1903c79-da4a-4d4a-9140-b5bb5b49133c
# ╠═a438af2e-ca31-4427-bc74-e84301d1f9cf
# ╠═d7e60d83-fa62-451f-99f9-126f8cd0e821
# ╠═5942cdba-6f0d-4128-8ce3-57826fafad0a
# ╠═81f415cd-aea3-4c84-aede-c0040f5ac28c
# ╠═6ebaa094-22b0-4e09-aa7f-eb3492dbf4f6
# ╠═b28859d1-7c4d-438d-a490-c6407365cd6a
# ╠═d7d217ce-2a0b-4e18-8aa8-8b5e3af08363
# ╠═b96306e1-2c38-4245-8d0a-87d503ab9df8
# ╟─930f7d64-29ef-4e7b-826d-66243e9724e9
# ╠═03fc1bf9-bcf5-44e9-9542-da9325639907
# ╠═bcc12d64-1eb8-4edb-b042-57e70ce3b641
# ╠═d9df62ae-47e7-4bde-907e-9eff4251c17f
# ╠═20061a57-ac97-409d-8a88-7decef927609
# ╠═4ba1e06b-f1b9-4faa-99e8-abd133a9052b
# ╠═92356131-5c26-4257-a148-2b9f49f2c9c6
# ╠═cae8d738-94f5-47ec-a7cf-4da51c069c75
# ╠═ac58190d-05bd-4e3e-9797-9bc1ee32b545
# ╠═15928993-d9fc-4934-a36a-90fb54a322ba
# ╠═a07bb87e-8321-4aa8-9555-ae33a51c5fa7
# ╠═cbb6c21f-81fe-4859-be21-2ff3759321fb
# ╠═cb24610c-2eac-4054-a17c-0ef4f3668f6c
# ╠═5721178c-e93e-4f2e-b39b-58ebdf9edfb8
# ╠═5800b6d5-8593-4f1d-95f6-9e3353f63eb2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
