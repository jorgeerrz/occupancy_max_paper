### A Pluto.jl notebook ###
# v0.19.27

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

# ╔═╡ a649da15-0815-438d-9bef-02c6d204656e
begin
	#using Pkg
	#Pkg.activate("../Project.toml")
	using Plots, Distributions, PlutoUI, Parameters
	using StatsPlots, DelimitedFiles, ZipFile
end

# ╔═╡ 25c56490-3b9c-4825-b91d-8b9e41fc0f6b
using LinearAlgebra

# ╔═╡ 99246057-6b37-4d85-ae77-2a4210fac365
using StatsBase

# ╔═╡ 422493c5-8a90-4e70-bd06-40f8e6b254f1
gr()

# ╔═╡ 76f77726-7776-4975-9f30-3887f13ae3e7
default(titlefont = ("Computer Modern",16), legend_font_family = "Computer Modern", legend_font_pointsize = 14, guidefont = ("Computer Modern", 14), tickfont = ("Computer Modern", 14))

# ╔═╡ 393eaf2d-e8fe-4675-a7e6-32d0fe9ac4e7
begin
	PlutoUI.TableOfContents(aside = true)
end

# ╔═╡ b4e7b585-261c-4044-87cc-cbf669768145
#Time steps for episode animations
max_t_anim = 300

# ╔═╡ 7feeec1a-7d7b-4220-917d-049f1e9b101b
md"# Grid world environment"

# ╔═╡ 7e68e560-45d8-4429-8bff-3a8229c8c84e
@with_kw struct environment
	γ::Float64 = 0.9
	sizex::Int64 = 11
	sizey::Int64 = 11
	sizeu::Int64 = 100
	xborders::Vector{Int64} = [0,sizex+1]
	yborders::Vector{Int64} = [0,sizey+1]
	#Location of obstacles
	obstacles
	obstaclesx
	obstaclesy
	#Array of vectors where rewards can be found
	reward_locations
	reward_mags
end

# ╔═╡ a41419bd-1859-4a08-8ce0-a5476e256284
@with_kw struct paras
	γ = 0.9
	η = 0.5
	α = 1
	β = 0
	constant_actions = true
end

# ╔═╡ 986f5441-9361-4074-a7f6-7affe650e555
params = paras(α = 1,constant_actions = false)

# ╔═╡ 194e91cb-b619-4908-aebd-3136107175b7
function adm_actions(s_state,u_state,env::environment, pars)
	out = Any[]
	moving_actions = [[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]]
	ids_actions = collect(1:length(moving_actions))
	#If agent is "alive"
	if u_state > 1
		#To check that agent does not start at harmful state
		#if transition_u(s_state,u_state,[0,0],env) != 1
			out = deepcopy(moving_actions)
			#When we constrain by hand the amount of actions
			#we delete some possible actions
			if pars.constant_actions == false
				#Give all possible actions by default
				#Check the boundaries of gridworld
				for it in 1:2
				not_admissible = in(env.xborders).(s_state[1]+(-1)^it)
					if not_admissible == true
						ids = findall(idx -> idx[1] == (-1)^it,out)
						deleteat!(out,ids)
						deleteat!(ids_actions,ids)
					end
				end
				for it in 1:2
				not_admissible = in(env.yborders).(s_state[2]+(-1)^it)
					if not_admissible == true
						ids = findall(idx -> idx[2] == (-1)^it,out)
						deleteat!(out,ids)
						deleteat!(ids_actions,ids)
					end
				end
				#Check for obstacles
				for action in moving_actions
					idx = findall(i -> i == s_state+action,env.obstacles)
					if length(idx) > 0
						idx2 = findfirst(y -> y == action,out)
						deleteat!(out,idx2)
						deleteat!(ids_actions,idx2)
					end
				end
			#end
		end
	else
		ids_actions = Any[]
	end
	#Doing nothing is always an admissible action
	push!(out,[0,0])
	push!(ids_actions,length(ids_actions)+1)
	out,ids_actions
	#Checking if having all actions at every state changes results
	#return [[1,0],[0,1],[-1,0],[0,-1],[0,0]],[1,2,3,4,5]
end

# ╔═╡ a46ced5b-2e58-40b2-8eb6-b4840043c055
md"## Functions for dynamic programming"

# ╔═╡ 9404080e-a52c-42f7-9abd-ea488bf7abc2
function reachable_states(s_state,a)
	#Deterministic environment
	s_prime = s_state + a
	[s_prime]
end

# ╔═╡ 8675158f-97fb-4222-a32b-49ce4f6f1d41
function transition_s(s,a,env)
	s_prime = s + a
	if in(env.obstacles).([s_prime]) == [true]
		s_prime = s
	else
		if s_prime[1] == env.xborders[1] || s_prime[1] == env.xborders[2] || s_prime[2] == env.yborders[1] || s_prime[2] == env.yborders[2]
			s_prime = s
		end
	end
	[s_prime]
end

# ╔═╡ 0dcdad0b-7acc-4fc4-93aa-f6eacc077cd3
function rewards(s,a,env::environment)
	rewards = 0
	s_p = transition_s(s,a,env)
	for i in 1:length(env.reward_locations)
		if s == env.reward_locations[i]
			rewards += env.reward_mags[i]
		end
	end
	action_cost = 0
	#Only deduct for action if action had an impact on the world
	#action_cost = (abs(a[1]) + abs(a[2]))
	#locs = findall(i -> equal(env.reward_locations[i], s_p))
	rewards - action_cost - 1
end

# ╔═╡ 0ce119b1-e269-41e2-80b7-991cae37cf5f
function transition_u(s,u,a,env)
	u_prime = u + rewards(s,a,env)
	if u_prime > env.sizeu
		u_prime = env.sizeu
	elseif u_prime <= 0
		u_prime = 1
	elseif u == 1
		u_prime = 1
	end
	u_prime
end

# ╔═╡ 92bca36b-1dc9-4c03-88c0-6a684dfbec9f
md"## Helper functions"

# ╔═╡ c96e3331-1dcd-4b9c-b28d-d74493c8934d
function build_index(s::Vector{Int64},u::Int64,env::environment)
	Int64(s[1] + env.sizex*(s[2]-1) + env.sizex*env.sizey*(u-1))
end

# ╔═╡ d0a5c0fe-895f-42d8-9db6-3b0fcc6bb43e
md" # Initialize 4 room grid world"

# ╔═╡ 155056b5-21ea-40d7-8cce-19fde5a1b150
begin
	tol = 1E-2
	n_iter = 2000
end

# ╔═╡ 6c716ad4-23c4-46f8-ba77-340029fcce87
function initialize_fourrooms(size_x,size_y,capacity,reward_locations,reward_mags) #
	#Four big rooms
	wall_x = Int(size_x+1)/2
	wall_y = Int(size_y+1)/2
	obstacles = Any[]
	for i in 1:size_x
		if i != Int((size_x+1)/4) && i != Int(3*(size_x+1)/4)
			#println("i = ", i)
			push!(obstacles, [i,wall_y])
		end
	end
	for j in 1:size_y
		if j != Int((size_y+1)/4) && j != Int(3*(size_y+1)/4)
			push!(obstacles, [wall_x,j])
		end
	end
	obstaclesx = Any[]
	obstaclesy = Any[]
	for i in 1:length(obstacles)
		push!(obstaclesx,obstacles[i][1])
		push!(obstaclesy,obstacles[i][2])
	end
environment(sizex = size_x, sizey = size_y, sizeu = capacity,obstacles = obstacles,obstaclesx = obstaclesx,obstaclesy = obstaclesy,reward_locations = reward_locations,reward_mags = reward_mags)
end

# ╔═╡ 07abd5b7-b465-425b-9823-19b73d07db56
@bind which PlutoUI.Select(["free" => "Free environment","wall" => "Environment with wall"], default = "free")

# ╔═╡ 8f2fdc23-1b82-4479-afe7-8eaf3304a122
begin
	size_x = 11
	size_y = 11
	capacity = 100
	food_gain = 10
	reward_locations = [[1,1],[size_x,size_y],[size_x,1],[1,size_y]]
	reward_mags = [1,1,1,1].*food_gain
	env1 = initialize_fourrooms(size_x,size_y,capacity,reward_locations,reward_mags)
	#one small room
	#obstacles = [[1,4],[2,4],[3,4],[4,3],[4,2],[4,1]]
	#Rewards for the walled environment
	#reward_locations = [[1,6],[3,2],[4,2],[5,2],[6,2]]
	#reward_mags = [30,-40,-30,-20,-10]
end

# ╔═╡ ec119eb7-894c-4eb1-95ea-4b4dae9526c3
transition_s([5,3],[1,1],env1),transition_u([5,3],10,[1,1],env1)

# ╔═╡ 403a06a7-e30f-4aa4-ade1-55dee37cd514
function draw_environment(x_pos,y_pos,u,env::environment,pars)
	reward_sizes = env.reward_mags
	#Draw obstacles
	ptest = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 20, color = "gray")
	#Draw food
	for i in 1:length(reward_mags)
		scatter!(ptest,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i]*2,color = "green",markershape = :diamond)
	end
	#Draw agent
	plot!(ptest, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	#Draw arrows
	actions,_ = adm_actions([x_pos[1],y_pos[1]],u[1],env,pars)
	arrow_x = zeros(length(actions))
	arrow_y = zeros(length(actions))
	aux = actions
	for i in 1:length(aux)
		mult = 1
		if norm(aux[i]) > 1
			mult = 1/sqrt(2)
		end
		arrow_x[i] = aux[i][1]*mult*1.3
		arrow_y[i] = aux[i][2]*mult*1.3
	end
	quiver!(ptest,ones(Int64,length(aux))*x_pos[1],ones(Int64,length(aux))*y_pos[1],quiver = (arrow_x,arrow_y),color = "black",linewidth = 3)
	#Draw agent
	ptest2 = plot(xticks = false,ylim = (0,env.sizeu), grid = false,legend = false)
	scatter!(ptest, x_pos,y_pos, markersize = 15, leg = false, color = "gray",markershape = :uptriangle)
	plot(ptest, size = (500,500),minorgrid = false)
	#Draw internal states
	#bar!(ptest2, u, color = "green")
	#plot(ptest,ptest2,layout = Plots.grid(1, 2, widths=[0.8,0.2]), title=["" "u(t)"], size = (700,500))
end

# ╔═╡ ac3a4aa3-1edf-467e-9a47-9f6d6655cd04
md"# H agent"

# ╔═╡ c6870051-0241-4cef-9e5b-bc876a3894fa
function h_iteration(env::environment,params;tolerance = 1E-2, n_iter = 100,verbose = false)
	value = zeros(env.sizex,env.sizey,env.sizeu)
	value_old = deepcopy(value)
	t_stop = n_iter
	f_error = 0
	for t in 1:n_iter
		ferror_max = 0
		Threads.@threads for u in 1:env.sizeu
			for x in 1:env.sizex, y in 1:env.sizey
				s = [x,y]
				ids = findall(i -> i == s,env.obstacles)
				#Only update value for non-obstacle states
				if length(ids) == 0
					actions,_ = adm_actions(s,u,env,params)
					Z = 0
					for a in actions
						s_primes = transition_s(s,a,env)
						expo = 0
						for s_prime in s_primes
							u_prime = transition_u(s,u,a,env)
							expo += env.γ*value_old[s_prime[1],s_prime[2],u_prime]
						end
						Z += exp(expo/params.α)
					end
					value[x,y,u] = params.α*log(Z)
					f_error = abs(value[x,y,u] - value_old[x,y,u])
					ferror_max = max(ferror_max,f_error)
				else
					value[x,y,u] = 0
				end
			end
		end
		if ferror_max < tolerance
			t_stop = t
			break
		end
		if verbose == true 
			println("iteration = ", t, ", max function error = ", ferror_max)
		end
		value_old = deepcopy(value)
	end
	value,t_stop
end

# ╔═╡ a6fbddc0-7def-489e-9ad1-2e6a5e68eddd
function optimal_policy(s,u,optimal_value,env::environment,params;verbose = false)
	actions,_ = adm_actions(s,u,env,params)
	policy = zeros(length(actions))
	Z = exp(optimal_value[s[1],s[2],u])
	#Only compute policy for available actions
	for (idx,a) in enumerate(actions)
		u_p = transition_u(s,u,a,env)
		s_p = transition_s(s,a,env)[1]
		policy[idx] = exp(env.γ*optimal_value[s_p[1],s_p[2],u_p]/params.α-optimal_value[s[1],s[2],u])
	end
	#adjust for numerical errors in probability
	sum_p = sum(policy)
	if verbose == true
		println("state = ", s, " u = ", u)
		println("policy = ", policy)
		println("sum policy = ", sum(policy))
	end
	policy = policy./sum(policy)
	actions,policy
end

# ╔═╡ d88e0e27-2354-43ad-9c26-cdc90beeea0f
# function optimal_policy(s,u,optimal_value,env::environment,verbose = false)
# 	actions,_ = adm_actions(s,u,env)
# 	policy = zeros(length(actions))
# 	Z = exp(optimal_value[s[1],s[2],u])
# 	#Only compute policy for available actions
# 	for (idx,a) in enumerate(actions)
# 		u_p = transition_u(s,u,a,env)
# 		s_p = transition_s(s,a,env)
# 		policy[idx] = exp(env.γ*optimal_value[s_p[1],s_p[2],u_p]-optimal_value[s[1],s[2],u])
# 	end
# 	#adjust for numerical errors in probability
# 	sum_p = sum(policy)
# 	if verbose == true
# 		println("state = ", s, " u = ", u)
# 		println("policy = ", policy)
# 		println("sum policy = ", sum(policy))
# 	end
# 	policy = policy./sum(policy)
# 	actions,policy
# end

# ╔═╡ 184636e2-c87d-4a89-b231-ff4aef8424d5
md"## Optimal value function"

# ╔═╡ 82fbe5a0-34a5-44c7-bdcb-36d16f09ea7b
begin
	#To compute
	env_c = initialize_fourrooms(size_x,size_y,50,reward_locations,reward_mags)
	h_value,t_stop = h_iteration(env1,params,tolerance = 0.1,n_iter = 30,verbose =true)
	#Specific one
	#h_value = reshape(readdlm("values/h_value_gain_$(food_gain).dat"),env1.sizex,env1.sizey,env1.sizeu)
end;

# ╔═╡ a11b198f-0a55-4529-b44c-270f37ef773a
#writedlm("h_value_u_$(env1.sizeu).dat",h_value)

# ╔═╡ 25e35560-5d80-4388-8002-fd29d0541b18
function arrow0!(x, y, u, v; as=0.07, lw=1, lc=:black, la=1)
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = as*nuv*v4, as*nuv*v5
    plot!([x,x+u], [y,y+v], lw=lw, lc=lc, la=la)
    plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], lw=lw, lc=lc, la=la)
    plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], lw=lw, lc=lc, la=la)
end

# ╔═╡ 73722c01-adee-4bfd-97b4-60f2ced23725
function plot_optimal_policy(p,u,opt_value,env::environment,constant_actions = false; arrow_length = 0.6,arrow_width = 1.4)
	for x in 1:env.sizex
		for y in 1:env.sizey 
			ids = findall(i -> i == [x,y],env.obstacles)
			if length(ids) == 0
				actions,probs = optimal_policy([x,y],u,opt_value,env,params,verbose = false)
				arrow_x = zeros(length(actions))
				arrow_y = zeros(length(actions))
				aux = actions.*probs
				for i in 1:length(aux)
					arrow_x[i] = aux[i][1]*arrow_length
					arrow_y[i] = aux[i][2]*arrow_length
				end
				arrow0!.(ones(Int64,length(aux))*x,ones(Int64,length(aux))*y,arrow_x,arrow_y,as = 0.4,lw = arrow_width)
				#quiver!(p,ones(Int64,length(aux))*x,ones(Int64,length(aux))*y,quiver = (arrow_x,arrow_y),color = "green",linewidth = 1)
				#scatter!(p,ones(Int64,length(aux))*x + arrow_x, ones(Int64,length(aux))*y + arrow_y,markersize = probs*10, color = "red")
				scatter!(p,[x + arrow_x[end]], [y + arrow_y[end]],markersize = probs[end]*15, color = "red")
			end
		end
	end
end		

# ╔═╡ e67db730-ca7c-4ef4-a2d2-7e001d5f7a79
u = 6
#@bind u PlutoUI.Slider(1:env_c.sizeu,default = 10)

# ╔═╡ 76f506dc-b21d-4e13-a8e8-9d1b3bd21b30
begin
	clim_v = (6,16)#(minimum(h_value[3,:,u]),maximum(h_value[3,:,u]))
		
	p1 = heatmap(transpose(h_value[:,:,u]), title = "\$V^*(x,y, u = $(u-1))\$",clims = clim_v)
	#Draw obstacles
	plot!(p1,env1.obstaclesx,env1.obstaclesy,bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)), st = :histogram2d, color = "#464646",colorbar = false)
	#Draw value
	for i in 1:length(env1.reward_mags)
		if reward_mags[i]  > 0
			col = "green"
		else
			col = "gray"
		end
		scatter!(p1,[env_c.reward_locations[i][1]],[env_c.reward_locations[i][2]], color = col, markersize = min(abs(env1.reward_mags[i]),50))
	end

	plot!(p1,size = (400,420)) 
	plot_optimal_policy(p1,u,h_value,env1,arrow_length = 0.8,arrow_width = 1.5)
	#p2 = heatmap(val_stoch[:,:,u], title = "value function random walk")
	plot(p1,legend = false,axis = false,dpi = 300)
	#savefig("optimal_value_function_deterministic.pdf")
end

# ╔═╡ aa5e5bf6-6504-4c01-bb36-df0d7306f9de
md"## Sample trajectory"

# ╔═╡ ef9e78e2-d61f-4940-9e62-40c6d060353b
function sample_trajectory(s_0,u_0,opt_value,max_t,env::environment,params,occupancies = false)
	xpositions = Any[]
	ypositions = Any[]
	u_states = Any[]
	all_x = Any[]
	all_y = Any[]
	urgency = Any[]
	dead_times = Any[]
	push!(xpositions,[s_0[1]])
	push!(ypositions,[s_0[2]])
	push!(all_x,s_0[1])
	push!(all_y,s_0[2])
	push!(u_states,u_0)
	unvisited_s_states = Any[]
	unvisited_u_states = collect(1:env.sizeu)
	n_arena_states = env.sizex*env.sizey - length(env.obstacles)
	t_max = max_t
	for x in 1:env.sizex
		for y in 1:env.sizey
			if ([x,y] in env.obstacles) == false
				push!(unvisited_s_states,[x,y])
			end
		end
	end
	s = deepcopy(s_0)
	u = deepcopy(u_0)
	id_s = findfirst(i -> i == [s[1],s[2]],unvisited_s_states)
	id_u = findfirst(i -> i == u,unvisited_u_states)
	if id_s != nothing
	deleteat!(unvisited_s_states,id_s)
	end
	if id_u != nothing
	deleteat!(unvisited_u_states,id_u)
	end
	for t in 1:max_t
		actions,policy = optimal_policy(s,u,opt_value,env,params)
		idx = rand(Categorical(policy))
		action = actions[idx]
		for i in 1:length(actions)
			if policy[i] > 0.6
				push!(urgency,u)
			end
		end
		s_p = transition_s(s,action,env)
		u_p = transition_u(s,u,action,env)
		s = s_p[1]
		u = u_p
		id_s = findfirst(i -> i == [s[1],s[2]],unvisited_s_states)
		id_u = findfirst(i -> i == u,unvisited_u_states)
		if id_s != nothing
			deleteat!(unvisited_s_states,id_s)
		end
		if id_u != nothing
			deleteat!(unvisited_u_states,id_u)
		end
		push!(xpositions,[s[1]])
		push!(ypositions,[s[2]])
		push!(all_x,s[1])
		push!(all_y,s[2])
		push!(u_states,u)
		if u == 1
			u = env.sizeu
			push!(dead_times,t+1)
		end
		if occupancies == true
			if length(unvisited_s_states) == 0
				return xpositions,ypositions,u_states,all_x,all_y,urgency,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,t
			end
		end
	end
	xpositions,ypositions,u_states,all_x,all_y,urgency,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,t_max,dead_times
end

# ╔═╡ a4457d71-27dc-4c93-81ff-f21b2dfed41d
md"### A movie"

# ╔═╡ 7ad00e90-3431-4e61-9a7f-efbc14d0724e
function animation(x_pos,y_pos,us,max_t,env::environment; title = "H agent", color = "blue")
anim = @animate for t in 1:max_t+2
	reward_sizes = env.reward_mags
	#Draw obstacles
	ptest = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 24, color = "black")
	#Draw food
	for i in 1:length(env.reward_mags)
		scatter!(ptest,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i],color = "green",markershape = :diamond)
	end
	plot!(ptest, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	ptest2 = plot(xticks = false,ylim = (0,env.sizeu), grid = false,legend = false)
	if t <= max_t
		scatter!(ptest, x_pos[t],y_pos[t], markersize = 15, leg = false, color = "gray",markershape = :utriangle)
	bar!(ptest2, [us[t]], color = color)
		if us[t] == 1
			scatter!(ptest,x_pos[t],y_pos[t],markersize = 30,markershape = :xcross,color = "black")
		end
	else 
		scatter!(ptest,[(env.sizex+1)/2],[(env.sizey+1)/2],markersize = 100,markershape = :square, color = "gray",leg = false)
	end
	plot(ptest,ptest2,layout = Plots.grid(1, 2, widths=[0.8,0.2]), title=[title "Energy"], size = (800,600))
end
end

# ╔═╡ a12ae2fc-07d4-459e-8459-747ab12fdbd5
h_xpos_anim,h_ypos_anim,h_us_anim,h_allx_anim,h_ally_anim,h_urgency_anim = sample_trajectory([3,3],30,h_value,max_t_anim,env1,params)

# ╔═╡ b072360a-6646-4d6d-90ea-716085c53f66
md"Produce animation? $(@bind movie CheckBox(default = false))"

# ╔═╡ f98d6ea0-9d98-4940-907c-455397158f3b
md"# R agent"

# ╔═╡ 5f4234f5-fc0e-4cdd-93ea-99b6463b2ba1
function reachable_rewards(s,u,a,env::environment;delta_reward = 1)
	r = 0
	if in(env.reward_locations).([s]) == [true] && u > 1
		r = delta_reward
	end
	r
end

# ╔═╡ 7a0173ac-240d-4f93-b413-45c6af0f4011
function q_iteration(env::environment,ϵ,pars,tolerance = 1E-2, n_iter = 100,verbose = false)
	value = zeros(env.sizex,env.sizey,env.sizeu)
	value_old = deepcopy(value)
	t_stop = n_iter
	f_error = 0
	for t in 1:n_iter
		ferror_max = 0
		Threads.@threads for u in 1:env.sizeu
			for x in 1:env.sizex, y in 1:env.sizey
				s = [x,y]
				ids = findall(i -> i == s,env.obstacles)
				#Only update value for non-obstacle states
				if length(ids) == 0
					actions,_ = adm_actions(s,u,env,pars)
					values = zeros(length(actions))
					for (id_a,a) in enumerate(actions)
						s_primes = transition_s(s,a,env)#reachable_states(s,a)
						r = reachable_rewards(s,u,a,env)
						for s_prime in s_primes
							u_prime = transition_u(s,u,a,env)
							values[id_a] += r + env.γ*value_old[s_prime[1],s_prime[2],u_prime]
						end
					end
					#value[x,y,u] = maximum(values)
					value[x,y,u] = (1-ϵ)*maximum(values) + (ϵ/length(actions))*sum(values)
					f_error = abs(value[x,y,u] - value_old[x,y,u])
					ferror_max = max(ferror_max,f_error)
				else
					value[x,y,u] = 0
				end
			end
		end
		if ferror_max < tolerance
			t_stop = t
			break
		end
		if verbose == true 
			println("iteration = ", t, ", max function error = ", ferror_max)
		end
		value_old = deepcopy(value)
	end
	value,t_stop
end

# ╔═╡ 27011f44-929a-4899-b822-539d270959e1
ϵ = 0.45

# ╔═╡ caadeb3b-0938-4559-8122-348c960a6eb1
#To compute
q_value,t_stop_q = q_iteration(env1,0.0,params,0.1,30,true);
#To read out from file
#Specific one
#q_value = reshape(readdlm("values/q_value_gain_$(food_gain)_eps_$(ϵ).dat"),env1.sizex,env1.sizey,env1.sizeu);

# ╔═╡ 29a4d235-8b03-4701-af89-cd289f212e7d
#writedlm("q_value_u_$(env1.sizeu).dat",q_value)

# ╔═╡ 819c1be2-339f-4c37-b8a3-9d8cb6be6496
u_q = 20

# ╔═╡ 358bc5ca-c1f6-40f1-ba2d-7e8466531903
begin
	p1_q = heatmap(transpose(q_value[:,:,u_q]), title = "optimal value function, u = $u_q",clims = (minimum(q_value[1,:,u_q]),maximum(q_value[1,:,u_q])))
	for i in 1:length(env1.reward_mags)
		if reward_mags[i]  > 0
			col = "green"
		else
			col = "gray"
		end
		scatter!(p1_q,[env1.reward_locations[i][1]],[env1.reward_locations[i][2]], color = col, markersize = min(abs(env1.reward_mags[i]),50))
	end
	plot!(p1_q,size = (1000,800)) 
	#plot_optimal_policy(p1,u,h_value,env1)
	#p2 = heatmap(val_stoch[:,:,u], title = "value function random walk")
	plot(p1_q,legend = false)
	#savefig("optimal_value_function.png")
end

# ╔═╡ 40d62df0-53bb-4b46-91b7-78ffd621a519
function optimal_policy_q(s,u,value,ϵ,env::environment,pars)
	actions,ids_actions = adm_actions(s,u,env,pars)
	q_values = zeros(length(actions))
	policy = zeros(length(actions))
	for (idx,a) in enumerate(actions)
		#s_primes = reachable_states(s,a)
		r = reachable_rewards(s,u,a,env)
		#for s_p in s_primes
			u_p = transition_u(s,u,a,env)
			s_p = transition_s(s,a,env)[1]
			#deterministic environment
			#u_p = transition_u(s,u,a,env)
			q_values[idx] += r + env.γ*value[s_p[1],s_p[2],u_p]
		#end
	end
	best_actions = findall(i-> i == maximum(q_values),q_values)
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

# ╔═╡ 005720d3-5920-476b-9f96-39971f512452
optimal_policy_q([3,3],20,q_value,ϵ,env1,params)

# ╔═╡ 2379dcc3-53cb-4fb6-b1e8-c851e36acd1f
function sample_trajectory_q(s_0,u_0,opt_value,ϵ,max_t,env::environment,pars,occupancies = false)
	xpositions = Any[]
	ypositions = Any[]
	u_states = Any[]
	all_x = Any[]
	all_y = Any[]
	push!(xpositions,[s_0[1]])
	push!(ypositions,[s_0[2]])
	push!(all_x,s_0[1])
	push!(all_y,s_0[2])
	push!(u_states,u_0)
	unvisited_s_states = Any[]
	unvisited_u_states = collect(1:env.sizeu)
	n_arena_states = env.sizex*env.sizey - length(env.obstacles)
	for x in 1:env.sizex
		for y in 1:env.sizey
			if ([x,y] in env.obstacles) == false
				push!(unvisited_s_states,[x,y])
			end
		end
	end
	s = deepcopy(s_0)
	u = deepcopy(u_0)
	id_s = findfirst(i -> i == [s[1],s[2]],unvisited_s_states)
	id_u = findfirst(i -> i == u,unvisited_u_states)
	deleteat!(unvisited_s_states,id_s)
	deleteat!(unvisited_u_states,id_u)
	for t in 1:max_t
		actions_at_s,policy,n_actions = optimal_policy_q(s,u,opt_value,ϵ,env,pars)
		idx = rand(Categorical(policy))
		action = actions_at_s[idx]
		#action = rand(actions)
		s_p = transition_s(s,action,env)
		u_p = transition_u(s,u,action,env)
		s = s_p[1]
		u = u_p
		#If agent dies, terminate episode
		if u == 1
			return xpositions,ypositions,u_states,all_x,all_y,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,t
		end
		id_s = findfirst(i -> i == [s[1],s[2]],unvisited_s_states)
		id_u = findfirst(i -> i == u,unvisited_u_states)
		if id_s != nothing
			deleteat!(unvisited_s_states,id_s)
		end
		if id_u != nothing
			deleteat!(unvisited_u_states,id_u)
		end
		push!(xpositions,[s[1]])
		push!(ypositions,[s[2]])
		push!(all_x,s[1])
		push!(all_y,s[2])
		push!(u_states,u)		
		if occupancies == true
			if length(unvisited_s_states) == 0
				return xpositions,ypositions,u_states,all_x,all_y,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,t
			end
		end
	end
	xpositions,ypositions,u_states,all_x,all_y,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,max_t
end

# ╔═╡ 6e7b6b2a-5489-4860-930e-47b7df014840
md"## Animation"

# ╔═╡ 2ed5904d-03a3-4999-a949-415d0cf47328
md"Produce animation? $(@bind movie_q CheckBox(default = false))"

# ╔═╡ 6a29cc32-6abf-41c1-b6e3-f4cb33b76f46
function animation2(x_posh,y_posh,ush,x_posr,y_posr,usr,max_t,env::environment)
anim = @animate for t in 1:max_t+2
	reward_sizes = env.reward_mags
	#Draw obstacles
	ptest = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 16, color = "black")
	ptest3 = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 16, color = "black")
	#Draw food
	for i in 1:length(reward_mags)
		scatter!(ptest,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i],color = "green",markershape = :diamond)
		scatter!(ptest3,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i],color = "green",markershape = :diamond)
	end
	plot!(ptest, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	plot!(ptest3, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	ptest2 = plot(xticks = false,ylim = (0,env.sizeu), grid = false,legend = false)
	ptest4 = plot(xticks = false,ylim = (0,env.sizeu), grid = false,legend = false)
	if t <= max_t
		scatter!(ptest, x_posh[t],y_posh[t], markersize = 15, leg = false, color = "gray",markershape = :utriangle)
		scatter!(ptest3, x_posr[t],y_posr[t], markersize = 15, leg = false, color = "gray",markershape = :utriangle)
	bar!(ptest2, [ush[t]], color = palette(:default)[1])
		bar!(ptest4, [usr[t]], color = palette(:default)[2])
		if ush[t] == 1
			scatter!(ptest,x_posh[t],y_posh[t],markersize = 30,markershape = :xcross,color = "black")
		end
		if usr[t] == 1
			scatter!(ptest3,x_posr[t],y_posr[t],markersize = 30,markershape = :xcross,color = "black")
		end
	else 
		scatter!(ptest,[(env.sizex+1)/2],[(env.sizey+1)/2],markersize = 100,markershape = :square, color = "gray",leg = false)
		scatter!(ptest3,[(env.sizex+1)/2],[(env.sizey+1)/2],markersize = 100,markershape = :square, color = "gray",leg = false)
	end
	plot(ptest,ptest2,ptest3,ptest4,layout = Plots.grid(1, 4, widths=[0.4,0.1,0.4,0.1]), title=["H agent" "Energy" "R agent, ϵ = $(ϵ)" "Energy"], size = (1200,400))
end
end

# ╔═╡ ba9a241e-93b7-42fd-b790-d6010e2435bd
md"# Fixed actions"

# ╔═╡ 39b00e87-db6e-441f-8265-de75148d6519
params_fa = paras(α = 1,constant_actions = true)

# ╔═╡ a6e21529-00eb-40b8-9f40-b8b26171eaad
h_value_fa,_ = h_iteration(env1,params_fa,tolerance = 0.1,n_iter = 30,verbose =false);

# ╔═╡ c8c70b3f-45d5-4d38-95af-85ae85d1233c
q_value_fa,_ = q_iteration(env1,0.45,params_fa,0.1,30,false);

# ╔═╡ a14df073-5b27-4bdb-b4c8-03927909b12e
function KL_iteration(env::environment,params;tolerance = 1E-2, n_iter = 100,verbose = false,fixed_default = false)
	value = zeros(env.sizex,env.sizey,env.sizeu)
	value_old = deepcopy(value)
	t_stop = n_iter
	f_error = 0
	for t in 1:n_iter
		ferror_max = 0
		Threads.@threads for u in 1:env.sizeu
			for x in 1:env.sizex, y in 1:env.sizey
				s = [x,y]
				ids = findall(i -> i == s,env.obstacles)
				#Only update value for non-obstacle states
				if length(ids) == 0
					actions,_ = adm_actions(s,u,env,params)
					Z = 0
					for a in actions
						s_primes = transition_s(s,a,env)
						expo = 0
						for s_prime in s_primes
							u_prime = transition_u(s,u,a,env)
							if fixed_default == false
								expo += -log(length(actions)) + env.γ*value_old[s_prime[1],s_prime[2],u_prime]
							else
								expo += -log(9) + env.γ*value_old[s_prime[1],s_prime[2],u_prime]
							end
						end
						Z += exp(expo/params.α)
					end
					value[x,y,u] = params.α*log(Z)
					f_error = abs(value[x,y,u] - value_old[x,y,u])
					ferror_max = max(ferror_max,f_error)
				else
					value[x,y,u] = 0
				end
			end
		end
		if ferror_max < tolerance
			t_stop = t
			break
		end
		if verbose == true 
			println("iteration = ", t, ", max function error = ", ferror_max)
		end
		value_old = deepcopy(value)
	end
	value,t_stop
end

# ╔═╡ ad2855ee-ffc3-4910-a307-2e908b93706b
axs,_ = adm_actions([3,3],50,env1,params_fa)

# ╔═╡ 5b3bbb53-d5f7-462b-8425-c8c6c4ff3865
length(axs)

# ╔═╡ 828b1bcd-a093-463c-bb2f-4c1b5359939c
optimal_policy([1,2],4,h_value_fa,env1,params_fa)

# ╔═╡ c7bc355d-d8dd-4b82-ac81-1b9a6d5f71e0
KL_value_va,_ = KL_iteration(env1,params,tolerance = 0.1,n_iter = 30,verbose =false);

# ╔═╡ af848419-950a-4827-bc4e-fc3e8facd1a8
KL_value_fa,_ = KL_iteration(env1,params_fa,tolerance = 0.1,n_iter = 30,verbose =false,fixed_default = true);

# ╔═╡ f405c26e-f911-4dea-9364-f751259acf43
function optimal_policyKL(s,u,optimal_value,env::environment,params;verbose = false,fixed_default = false)
	actions,_ = adm_actions(s,u,env,params)
	policy = zeros(length(actions))
	Z = exp(optimal_value[s[1],s[2],u])
	#Only compute policy for available actions
	for (idx,a) in enumerate(actions)
		u_p = transition_u(s,u,a,env)
		s_p = transition_s(s,a,env)[1]
		if fixed_default == false
			policy[idx] = exp((-log(length(actions))+env.γ*optimal_value[s_p[1],s_p[2],u_p])/params.α-optimal_value[s[1],s[2],u])
		else
			policy[idx] = exp((-log(9) + env.γ*optimal_value[s_p[1],s_p[2],u_p])/params.α-optimal_value[s[1],s[2],u])
		end
	end
	#adjust for numerical errors in probability
	sum_p = sum(policy)
	if verbose == true
		println("state = ", s, " u = ", u)
		println("policy = ", policy)
		println("sum policy = ", sum(policy))
	end
	policy = policy./sum(policy)
	actions,policy
end

# ╔═╡ 91826a58-ebd5-4e2f-8836-ad6cff22406e
optimal_policyKL([1,2],4,KL_value_fa,env1,params_fa,fixed_default = true)

# ╔═╡ cb3280eb-8453-4040-aac0-22c1dc93b9d9
function create_episode_KL(s_0,u_0,value,max_t,env,pars;fd = false)
	states = Any[]
	us = Any[]
	values = Any[]
	a_s = Any[]
	s = deepcopy(s_0)
	u = deepcopy(u_0)
	for t in 1:max_t
		push!(states,s)
		push!(us,u)
		if u == 1
			break
		end
		#ids_dstate,discretized_state = discretize_state(state,env)
		#id_dstate = build_index_b(ids_dstate,env)
		push!(values,value[s[1],s[2],u])
		actions, policy= optimal_policyKL(s,u,value,env,pars,fixed_default = true)
		idx = rand(Categorical(policy))
		action = actions[idx]
		#Choosing action randomly according to policy
		a = action#rand(actions)
		push!(a_s,a)
		#For discretized dynamics
		#_,state_p = transition_b(state,action,env) 
		#For real dynamics
		states_p = transition_s(s,a,env)
		u_prime = transition_u(s,u,a,env)
		s = deepcopy(states_p[1])
		u = deepcopy(u_prime)
	#end
	end
	states,us,a_s,values
end

# ╔═╡ 7431c898-bbef-422e-af13-480d2e912d16
md"# Empowerment"

# ╔═╡ 86685690-c31d-4b71-b070-43fdb18e48db
function adm_actions_emp(s_state,u_state,env::environment, constant_actions = false)
	out = Any[]
	moving_actions = [[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]]
	#moving_actions = [[1,0],[0,1],[-1,0],[0,-1]]
	ids_actions = collect(1:length(moving_actions))
	#If agent is "alive"
	if u_state > 1
		#To check that agent does not start at harmful state
		#if transition_u(s_state,u_state,[0,0],env) != 1
			out = deepcopy(moving_actions)
			#When we constrain by hand the amount of actions
			#we delete some possible actions
			if constant_actions == false
				#Give all possible actions by default
				#Check the boundaries of gridworld
				for it in 1:2
				not_admissible = in(env.xborders).(s_state[1]+(-1)^it)
					if not_admissible == true
						ids = findall(idx -> idx[1] == (-1)^it,out)
						deleteat!(out,ids)
						deleteat!(ids_actions,ids)
					end
				end
				for it in 1:2
				not_admissible = in(env.yborders).(s_state[2]+(-1)^it)
					if not_admissible == true
						ids = findall(idx -> idx[2] == (-1)^it,out)
						deleteat!(out,ids)
						deleteat!(ids_actions,ids)
					end
				end
				#Check for obstacles
				for action in moving_actions
					idx = findall(i -> i == s_state+action,env.obstacles)
					if length(idx) > 0
						idx2 = findfirst(y -> y == action,out)
						deleteat!(out,idx2)
						deleteat!(ids_actions,idx2)
					end
				end
			#end
		end
	else
		ids_actions = Any[]
	end
	#Doing nothing is always an admissible action
	push!(out,[0,0])
	push!(ids_actions,length(ids_actions)+1)
	out,ids_actions
	#Checking if having all actions at every state changes results
	#return [[1,0],[0,1],[-1,0],[0,-1],[0,0]],[1,2,3,4,5]
end

# ╔═╡ f2f45f6f-c973-4272-afa2-7077dd169790
@with_kw struct pars_emp
	n_fixed = 1 #fixing the action for n_fixed simulation steps
	n = 3 #n in n-step empowerment
	tol = 1E-12
	max_iter = 150
	constant_actions = false
end

# ╔═╡ 0de63f0b-c545-445b-a0da-97ec75647598
p_emp = pars_emp(n = 5)

# ╔═╡ c555b1ac-ed4f-4767-9680-0e7ef7ab758f
begin
	size_x_emp = 11
	size_y_emp = 11
	capacity_emp = 100
	food_gain_emp = 10
	reward_locations_emp = [[1,1],[size_x_emp,size_y_emp],[size_x_emp,1],[1,size_y_emp]]
	reward_mags_emp = [1,1,1,1].*food_gain_emp
	env_emp = initialize_fourrooms(size_x_emp,size_y_emp,capacity_emp,reward_locations_emp,reward_mags_emp)
	#env_emp = env1
	#one small room
	#obstacles = [[1,4],[2,4],[3,4],[4,3],[4,2],[4,1]]
	#Rewards for the walled environment
	#reward_locations = [[1,6],[3,2],[4,2],[5,2],[6,2]]
	#reward_mags = [30,-40,-30,-20,-10]
end

# ╔═╡ 33df863c-bf8c-4624-9fd3-1d3a569a4f61
function transition_s_emp(s,u,a,env)
	s_prime = [1,1]
	if u > 1
		s_prime = s + a
		if in(env.obstacles).([s_prime]) == [true]
			s_prime = s
		else
			if s_prime[1] == env.xborders[1] || s_prime[1] == env.xborders[2] || s_prime[2] == env.yborders[1] || s_prime[2] == env.yborders[2]
				s_prime = s
			end
		end
	end
	s_prime
end

# ╔═╡ 01aaed4b-e028-49c9-90b4-a6c4158929f1
function n_step_transition(s,u,n_action,env,pars_emp)
	s_temp = deepcopy(s)
	u_temp = deepcopy(u)
	for a in n_action #n-step action
		for t in 1:pars_emp.n_fixed #fix each action for n_fixed simulation steps
			u_p = transition_u(s_temp,u_temp,a,env)
			s_p = transition_s_emp(s_temp,u_p,a,env)
			s_temp = deepcopy(s_p)
			u_temp = deepcopy(u_p)
		end
	end
	s_temp,u_temp
end

# ╔═╡ 49df9fba-22a8-48e9-8278-22ffdfc7765f
s_p = n_step_transition([1,1],1,[[-1,1],[0,0]],env1,p_emp)

# ╔═╡ 3dd47db4-2485-4833-8172-3fe7a3fe2214
i_p = build_index(s_p[1],s_p[2],env1)

# ╔═╡ e4dcc71c-4cc6-4b8f-88ad-7c2c18e5af02
function empowerment(s_state,u_state,env,pars_emp)
	actions,_ = adm_actions_emp(s_state,u_state,env,pars_emp.constant_actions)
	n_step_actions = Iterators.product([actions for i in 1:pars_emp.n]...)
	N_n = length(actions)^pars_emp.n
	p_a = ones(N_n)*(1/N_n) #Initialize uniformly random prob of n-step actions
	#Create actions - states matrix
	n_s = env.sizex*env.sizey*env.sizeu
	P_a_s_big = zeros(N_n,n_s) #deterministic world, at most N_n distinct states
	#build matrix of n_actions to states
	for (ν,a) in enumerate(n_step_actions)
		s_p,u_p = n_step_transition(s_state,u_state,a,env,pars_emp)
		i_p = build_index(s_p,u_p,env)
		P_a_s_big[ν,i_p] = 1
	end
	#Keep a lower dimensional matrix, of only reachable states
	test = Any[]
	for i in 1:env_emp.sizex*env_emp.sizey*env_emp.sizeu
		if sum(P_a_s_big[:,i]) > 0
			push!(test,P_a_s_big[:, i])
		end
	end
	P_a_s = hcat(test...)
	#@show size(P_a_s)
	#Blahut-Arimoto algorithm
	c_old = 0
	q = zeros(N_n,n_s)
	for k in 1:pars_emp.max_iter
		c_new = 0
		p_k = zeros(N_n)
		q = p_a.*P_a_s
		q = q./sum(q, dims = 1)
		p_k = prod(q.^P_a_s, dims = 2)
		p_k = p_k/sum(p_k)
		# for (ν,a) in enumerate(n_step_actions)
		# 	if p_k[ν] > 0
		# 		c_new += sum(p_k[ν].*P_a_s[ν,:].*log.(q[ν,:]./p_k[ν] .+ 1E-16))
		# 	end
		# end
		#@show c_new,c_old
		if norm(p_k - p_a) < pars_emp.tol
		#if abs(c_new - c_old) < pars_emp.tol && k > 1
			#println("Converged at iteration $(k)")
			break
		end
		p_a = deepcopy(p_k)
		c_old = deepcopy(c_new)
	end
	#@show q
	for (ν,a) in enumerate(n_step_actions)
		if p_a[ν] > 0
			c_old += sum(p_a[ν].*P_a_s[ν,:].*log.(q[ν,:]./p_a[ν] .+ 1E-16))
		end
	end
	c_old, p_a
end

# ╔═╡ 4121c28e-010e-428a-8e9a-544231dc6c0b
empowerment([3,5],30,env_emp,p_emp)

# ╔═╡ 1c5176b2-f7e0-4a9a-b07c-392840294aa0
empowerment([2,2],30,env_emp,p_emp)

# ╔═╡ e2da5d6b-4134-4bee-a93b-a153d19359cf
function emp_episode(s_0,u_0,env,pars_emp;n_steps = 10)
	s = deepcopy(s_0)
	u = deepcopy(u_0)
	states_out = Any[]
	us_out = Any[]
	actions_out = Any[]
	empowerments_out = Any[]
	push!(states_out,s)
	push!(us_out,u)
	for t in 1:n_steps
		actions,_ = adm_actions(s,u,env,pars_emp)
		#n_step_actions = Iterators.product([actions for i in 1:pars_emp.n]...)
		#N_n = length(actions)^pars_emp.n
		#emps = zeros(N_n)
		emps = zeros(length(actions))
		#for every 1-step action, compute n-step empowerment of every successor state
		Threads.@threads for ν in 1:length(actions)
			u_p = transition_u(s,u,actions[ν],env)
			s_p = transition_s_emp(s,u_p,actions[ν],env)
			emps[ν],_ = empowerment(s_p,u_p,env,pars_emp)
		end
		emps = round.(emps,digits = 12)
		#take 1-step action that greedily maximizes n-step empowerment
		best_actions = findall(i-> i == maximum(emps),emps)
		#e_max,ν_max = findmax(emps)
		ν_max = rand(best_actions)
		@show t, actions, emps, best_actions
		a_max = actions[ν_max]
		e_max = emps[ν_max]
		u_p = transition_u(s,u,a_max,env)
		s_p = transition_s_emp(s,u_p,a_max,env)
		push!(actions_out,a_max)
		push!(empowerments_out, e_max)
		push!(states_out,s_p)
		push!(us_out,u)
		if u_p < 2
			break
		end
		s = deepcopy(s_p)
		u = deepcopy(u_p)
	end
	states_out,us_out,actions_out,empowerments_out
end

# ╔═╡ e46e0035-c649-485f-9e87-6f7231c0927a
# states_emp,us_emp, actions_emp,emp_emp = emp_episode([3,6],50,env_emp,p_emp,n_steps = 200)

# ╔═╡ b1737188-4e57-4d81-a1fa-b55e1621a7dc
# begin
# 	writedlm("empowerment/11x11_states_$(p_emp.n)_step_time_$(length(states_emp))_const_actions_$(p_emp.constant_actions)_gain_$(food_gain_emp).dat",[[states_emp[i],us_emp[i]] for i in 1:length(states_emp)])
# 	writedlm("empowerment/11x11_actions_$(p_emp.n)_step_time_$(length(states_emp))_const_actions_$(p_emp.constant_actions)_gain_$(food_gain_emp).dat",actions_emp)
# 	writedlm("empowerment/11x11_empowerments_$(p_emp.n)_step_time_$(length(states_emp))_const_actions_$(p_emp.constant_actions)_gain_$(food_gain_emp).dat",emp_emp)
# end

# ╔═╡ 73b6c6e3-0c3c-46d5-a109-1f512b6871ee
states_emp_read = readdlm("empowerment/11x11_states_$(p_emp.n)_step_time_201_const_actions_$(p_emp.constant_actions)_gain_$(food_gain_emp).dat")

# ╔═╡ 8ec4f121-410a-49c6-8ae5-f8e014757fc7
begin
	states_emp = [[parse(Int,states_emp_read[i,1][2]),parse(Int,states_emp_read[i,1][2])] for i in 1:length(states_emp_read[:,1])]
	us_emp = [states_emp_read[i,3] for i in 1:length(states_emp_read[:,1])]
end

# ╔═╡ 0a693ea2-9506-4217-86d7-514678c03104
anim_emp = animation([[states_emp[i][1]] for i in 1:length(states_emp)],[[states_emp[i][2]] for i in 1:length(states_emp)],us_emp,length(states_emp),env_emp,color = palette(:default)[1],title = "$(p_emp.n)-step empowerment")

# ╔═╡ 118e6de3-b525-42e5-8c0c-654c525a4684
gif(anim_emp, fps = 10, "grid_world_11x11_emp_n_$(p_emp.n)_t_$(length(states_emp))_const_actions_$(p_emp.constant_actions)_gain_$(food_gain_emp).gif")

# ╔═╡ 3014c91c-b705-42e1-9689-ebb1c357f8b2
md"# Sophisticated active inference"

# ╔═╡ 49124160-32b0-4e09-904c-9547697cbfd1
begin
	size_x_AIF = 11
	size_y_AIF = 11
	capacity_AIF = 100
	food_gain_AIF = 10
	reward_locations_AIF = [[1,1],[size_x_AIF,size_y_AIF],[size_x_AIF,1],[1,size_y_emp]]
	reward_mags_AIF = [1,1,1,1].*food_gain_AIF
	env_AIF = initialize_fourrooms(size_x_AIF,size_y_AIF,capacity_AIF,reward_locations_AIF,reward_mags_AIF)
	#env_emp = env1
	#one small room
	#obstacles = [[1,4],[2,4],[3,4],[4,3],[4,2],[4,1]]
	#Rewards for the walled environment
	#reward_locations = [[1,6],[3,2],[4,2],[5,2],[6,2]]
	#reward_mags = [30,-40,-30,-20,-10]
end

# ╔═╡ 521717ad-23b7-4bca-bc8d-a2387946e55d
@with_kw struct pars_AIF
	H = 10
	δ = 0.2
	#Reward maximizing temperature
	β = 1
	constant_actions = false
end

# ╔═╡ d99990c1-1b0c-4d52-87a3-6b3bd363fde7


# ╔═╡ 00025055-7ec0-486d-bb29-8e769e08fcc8
function AIF_iteration(env, pars)
	value = zeros(env.sizex,env.sizey,env.sizeu)
	value_old = deepcopy(value)
	for t in pars.H-1:-1:1
		#Parallelization over states
		Threads.@threads for u in 1:env.sizeu
			for x in 1:env.sizex, y in 1:env.sizey
				s = [x,y]
				ids = findall(i -> i == s,env.obstacles)
				#Only update value for non-obstacle states
				if length(ids) == 0
					actions,_ = adm_actions(s,u,env,pars)
					Q = zeros(length(actions))
					for (id_a,a) in enumerate(actions)
						#For every action, look at reachable states
						#s_primes_ids,states_p = reachable_states_b(state,a,env)
						s_primes = transition_s(s,a,env)
						for (idx,s_p) in enumerate(s_primes)
							u_prime = transition_u(s,u,a,env)
							#if at the end of the horizon
							if t == pars.H -1
								if u_prime == 1
									Q[id_a] -= 1
								else 
									Q[id_a] += pars.β*(reachable_rewards(s,u,a,env) + 1) # βR(s) + c(β)
									#Q[id_a] += pars.δ
								end
							else
								if u_prime == 1
									Q[id_a] -= 1
								else
									Q[id_a] += pars.β*(reachable_rewards(s,u,a,env) + 1) + value_old[s_p[1],s_p[2],u_prime]
									#Q[id_a] += pars.δ + value_old[s_p[1],s_p[2],u_prime]
								end
							end
						end
					end
					value[x,y,u],i_opt = findmax(Q)
				else
					value[x,y,u] = 0
				end
				
				#a_opt = actions[i_opt]
			end
		end
		value_old = deepcopy(value)
	end
	value
end

# ╔═╡ b7c66258-bb37-4660-9b22-783a50d4c6f8
function optimal_policy_AIF(s,u,value,env,pars)
	#Check admissible actions
	actions,ids_actions = adm_actions(s,u,env,pars)
	policy = zeros(length(actions))
	Q = zeros(length(actions))
	for (id_a,a) in enumerate(actions)
		#For every action, look at reachable states
		#s_primes_ids,states_p = reachable_states_b(state,a,env)
		states_p =  transition_s(s,a,env)
		for (idx,s_p) in enumerate(states_p)
			u_prime = transition_u(s,u,a,env)
			if u_prime == 1
				Q[id_a] += 0
			else
				Q[id_a] += pars.β*(reachable_rewards(s,u,a,env) + 1) + value[s_p[1],s_p[2],u_prime]
				#Q[id_a] += pars.δ + value[s_p[1],s_p[2],u_prime]
			end
		end
	end
	#computer precision
	#best_actions = findall(i-> i == maximum(round.(Q,digits = 16)),round.(Q,digits = 16))
	best_actions = findall(i-> i == maximum(Q),Q)
	actions[best_actions]
end

# ╔═╡ 4f6cdf92-0821-4d39-ae15-1844a4a29482
function create_episode_AIF(s_0,u_0,value,max_t,env,pars)
	states = Any[]
	us = Any[]
	values = Any[]
	a_s = Any[]
	s = deepcopy(s_0)
	u = deepcopy(u_0)
	for t in 1:max_t
		push!(states,s)
		push!(us,u)
		if u == 1
			break
		end
		#ids_dstate,discretized_state = discretize_state(state,env)
		#id_dstate = build_index_b(ids_dstate,env)
		push!(values,value[s[1],s[2],u])
		actions = optimal_policy_AIF(s,u,value,env,pars)
		#Choosing action randomly according to policy
		a = rand(actions)
		push!(a_s,a)
		#For discretized dynamics
		#_,state_p = transition_b(state,action,env) 
		#For real dynamics
		states_p = transition_s(s,a,env)
		u_prime = transition_u(s,u,a,env)
		s = deepcopy(states_p[1])
		u = deepcopy(u_prime)
	#end
	end
	states,us,a_s,values
end

# ╔═╡ 6edeefb3-186c-447a-b6e6-abab6a7d1c63
p_AIF = pars_AIF(H = 200,δ = 0.1,β = 0.1)

# ╔═╡ 317fab4f-c0b0-4671-a653-1bc8cbbfabc4
horizons = [20,30,50,100,200,500,1000]

# ╔═╡ 29193a6f-737f-4efb-acaa-d16d2b8c3589
βs_sAIF = [0.1]

# ╔═╡ 0137a63c-6beb-4aaa-a485-f4b5e08fab01
# begin
# 	# for h in horizons
# 	h = 200
# 	for β in βs_sAIF
# 		p_AIFt = pars_AIF(H = h,δ = 0.1,β = β)
# 		v_AIF = AIF_iteration(env_AIF,p_AIFt);
# 		writedlm("AIF/values/rew_beta_$(p_AIFt.β)_horizon_$(h).dat",v_AIF)
# 	end
# end

# ╔═╡ 210ef48a-13b1-44b5-a4c6-c2084bafc778
begin
	v_AIF = readdlm("AIF/values/rew_beta_$(p_AIF.β)_horizon_$(p_AIF.H).dat")
	v_AIF = reshape(v_AIF,env_AIF.sizex,env_AIF.sizey,env_AIF.sizeu)
end

# ╔═╡ 38d916de-9068-4f37-9ace-81e115d731bc
states_AIF_anim,us_AIF_anim,actions_AIF_anim,values_AIF_anim = create_episode_AIF([3,3],30,v_AIF,max_t_anim,env_AIF,p_AIF)

# ╔═╡ 6046a1ae-ec46-490c-baa7-1534f72c5ea9
anim_AIF = animation([[states_AIF_anim[i][1]] for i in 1:length(states_AIF_anim)],[[states_AIF_anim[i][2]] for i in 1:length(states_AIF_anim)],us_AIF_anim,length(states_AIF_anim),env_AIF,color = palette(:default)[1],title = "H = $(p_AIF.H) EFE agent, \$\\beta = $(p_AIF.β)\$")

# ╔═╡ 9ec3d452-5cc0-447d-8965-9b487f39650a
gif(anim_AIF, fps = 10, "AIF/rew_beta_$(p_AIF.β)_h_$(p_AIF.H).gif")

# ╔═╡ 51432185-5447-45a5-8734-f85ad54b5602
function animation3(x_pos1,y_pos1,us1,x_pos2,y_pos2,us2,x_pos3,y_pos3,us3,max_t,env::environment;titles = ["MOP", "Energy","5-step MPOW", "Energy","H = 200, \$\\lambda\$ = 0.1 EFE", "Energy"])
anim = @animate for t in 1:max_t+2
	reward_sizes = env.reward_mags
	#Draw obstacles
	ptest = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 18, color = "black")
	ptest3 = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 18, color = "black")
	ptest5 = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 18, color = "black")
	#Draw food
	for i in 1:length(reward_mags)
		scatter!(ptest,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i],color = "green",markershape = :diamond)
		scatter!(ptest3,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i],color = "green",markershape = :diamond)
		scatter!(ptest5,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i],color = "green",markershape = :diamond)
	end
	plot!(ptest, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	plot!(ptest3, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	plot!(ptest5, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	ptest2 = plot(xticks = false,ylim = (0,env.sizeu), grid = false,legend = false)
	ptest4 = plot(xticks = false,ylim = (0,env.sizeu), grid = false,legend = false)
	ptest6 = plot(xticks = false,ylim = (0,env.sizeu), grid = false,legend = false)
	if t <= max_t
		scatter!(ptest, x_pos1[t],y_pos1[t], markersize = 15, leg = false, color = "gray",markershape = :utriangle)
		scatter!(ptest3, x_pos2[t],y_pos2[t], markersize = 15, leg = false, color = "gray",markershape = :utriangle)
		scatter!(ptest5, x_pos3[t],y_pos3[t], markersize = 15, leg = false, color = "gray",markershape = :utriangle)
		bar!(ptest2, [us1[t]], color = palette(:default)[1])
		bar!(ptest4, [us2[t]], color = palette(:default)[2])
		bar!(ptest6, [us3[t]], color = palette(:default)[3])
		if us1[t] == 1
			scatter!(ptest,x_pos1[t],y_pos1[t],markersize = 30,markershape = :xcross,color = "black")
		end
		if us2[t] == 1
			scatter!(ptest3,x_pos2[t],y_pos2[t],markersize = 30,markershape = :xcross,color = "black")
		end
		if us3[t] == 1
			scatter!(ptest5,x_pos3[t],y_pos3[t],markersize = 30,markershape = :xcross,color = "black")
		end
	else 
		scatter!(ptest,[(env.sizex+1)/2],[(env.sizey+1)/2],markersize = 100,markershape = :square, color = "gray",leg = false)
		scatter!(ptest3,[(env.sizex+1)/2],[(env.sizey+1)/2],markersize = 100,markershape = :square, color = "gray",leg = false)
		scatter!(ptest5,[(env.sizex+1)/2],[(env.sizey+1)/2],markersize = 100,markershape = :square, color = "gray",leg = false)
	end
	plot(ptest,ptest2,ptest3,ptest4,ptest5,ptest6,layout = Plots.grid(1, 6, widths=[0.3,0.03,0.3,0.03,0.3,0.03]),margin = 4Plots.mm, title=[titles[1] titles[2] titles[3] titles[4] titles[5] titles[6]], size = (1800,450))
end
end

# ╔═╡ 48ade83b-bd73-4570-9a14-590f6f846097
anim3 = animation3(h_xpos_anim,h_ypos_anim,h_us_anim,[[states_emp[i][1]] for i in 1:length(states_emp)],[[states_emp[i][2]] for i in 1:length(states_emp)],us_emp,[[states_AIF_anim[i][1]] for i in 1:length(states_AIF_anim)],[[states_AIF_anim[i][2]] for i in 1:length(states_AIF_anim)],us_AIF_anim,200,env1)

# ╔═╡ d51573c2-f8b8-4785-bc14-3f611a34a924
us_emp

# ╔═╡ c515caee-858d-4e1f-b021-7dd5b8648549
gif(anim3,fps = 10,"MOP_MPOW_EFE.gif")

# ╔═╡ df4ce514-884c-401a-85f3-02ec37444816
md"## Long episode analysis"

# ╔═╡ 446c2897-8db9-452d-9d11-4cf7cc9bfa8a
num_episodes_AIF = 1000

# ╔═╡ 89c24ce1-87ce-4145-b3e7-bc6da50b94bf
max_time_AIF = 10000

# ╔═╡ 8dcb6536-3737-4d58-862e-094218874040
env_AIF

# ╔═╡ 4d96f296-b22a-4243-b32e-7154d88a16d7
β_survivals = 0.1

# ╔═╡ a8152784-c97f-4937-8791-166a411eee80
# begin
# 	survival_pcts_AIF = zeros(length(horizons),num_episodes_AIF)
# 	entropies_AIF = zeros(length(horizons),num_episodes_AIF)
# 	bins = (collect(1:env_AIF.sizex),collect(1:env_AIF.sizey),collect(1:env_AIF.sizeu))
# 	for (i,h) in enumerate(horizons)
# 		for j in 1:num_episodes_AIF
# 			p_AIFt = pars_AIF(H = h,δ = 0.1,β = β_survivals)
# 			val_h = readdlm("AIF/values/horizon_$(h).dat")
# 			val_h = reshape(val_h,env_AIF.sizex,env_AIF.sizey,env_AIF.sizeu)
# 			states_AIF,us_AIF,actions_AIF,values_AIF = create_episode_AIF(s_0,50,val_h,max_time_AIF,env_AIF,p_AIFt)
# 			h_AIF = fit(Histogram, ([states_AIF[i][1] for i in 1:length(states_AIF)],[states_AIF[i][2] for i in 1:length(states_AIF)],us_AIF), bins)
# 			h_AIF_n = normalize(h_AIF,mode =:probability)
# 			entropies_AIF[i,j] = entropy(h_AIF_n.weights)
# 			survival_pcts_AIF[i,j] = length(states_AIF)
# 		end
# 	end
# end

# ╔═╡ e9b5ab32-0899-4e3d-b1cb-c986ae82b12d
# begin
# 	writedlm("AIF/survivals/suvival_pcts_rew_beta_$(β_survivals)_horizons_$(horizons).dat",survival_pcts_AIF)
# 	writedlm("AIF/survivals/entropies_rew_beta_$(β_survivals)_horizons_$(horizons).dat",entropies_AIF)
# end

# ╔═╡ b8ce5e63-5f88-424f-8d78-0910b0f88762
begin
	survival_pcts_AIF = readdlm("AIF/survivals/suvival_pcts_rew_beta_$(β_survivals)_horizons_$(horizons).dat")
	entropies_AIF = readdlm("AIF/survivals/entropies_rew_beta_$(β_survivals)_horizons_$(horizons).dat")
end

# ╔═╡ a12232f3-0c6b-44c2-bcba-f150a1f39c88
entropies_AIF

# ╔═╡ b92cde2a-bc29-4417-8a88-0f527a1b790e
begin
	plot(xlabel = "Horizon",ylabel = "Lifetime (steps)")
		plot!(horizons, mean(survival_pcts_AIF,dims = 2),yerror = std(survival_pcts_AIF,dims = 2)./(sqrt(length(survival_pcts_AIF[1,:]))),markerstrokewidth = 2, linewidth = 2.5,label = "AIF agent")
	#plot!(horizons, mean(survival_H).*ones(length(horizons_j)),label = "MOP agent")
	plot!(legend_position = :topright)
	#savefig("AIF/lifetimes_AIF.pdf")
end

# ╔═╡ d801413b-adff-48f0-aa90-89a1af1c0d63
md"# MaxEnt RL agent"

# ╔═╡ 2581ecba-d9c1-4989-b718-4f559c870adb
function maxent_iteration(env::environment,params;temp = 1,tolerance = 1E-2, n_iter = 100,verbose = false)
	value = zeros(env.sizex,env.sizey,env.sizeu)
	value_old = deepcopy(value)
	t_stop = n_iter
	f_error = 0
	for t in 1:n_iter
		ferror_max = 0
		Threads.@threads for u in 1:env.sizeu
			for x in 1:env.sizex, y in 1:env.sizey
				s = [x,y]
				ids = findall(i -> i == s,env.obstacles)
				#Only update value for non-obstacle states
				if length(ids) == 0
					actions,_ = adm_actions(s,u,env,params)
					Z = 0
					for a in actions
						reward = reachable_rewards(s,u,a,env)
						s_primes = transition_s(s,a,env)#reachable_states(s,a)
						exponent = 0
						for s_prime in s_primes
							u_prime = transition_u(s,u,a,env)
							exponent += env.γ*value_old[s_prime[1],s_prime[2],u_prime]
						end
						Z += exp((reward + exponent)/params.α)
					end
					value[x,y,u] = params.α*log(Z)
					f_error = abs(value[x,y,u] - value_old[x,y,u])
					ferror_max = max(ferror_max,f_error)
				else
					value[x,y,u] = 0
				end
			end
		end
		if verbose == true 
			println("iteration = ", t, ", max function error = ", ferror_max)
		end
		if ferror_max < tolerance
			t_stop = t
			break
		end

		value_old = deepcopy(value)
	end
	value,t_stop
end

# ╔═╡ 93980ec8-f9d2-4637-945c-259715e3ef5d
maxent_value, t_maxent = maxent_iteration(env1,params,tolerance = 0.1,n_iter = 50,verbose = true);

# ╔═╡ 0fdb6f27-479b-4b46-bea2-2b6158f3d1c7
md"# Toy problem"

# ╔═╡ 164985f1-3dda-4ab1-b3d2-af827eace611
toy_arena = environment(γ = 0.9,sizex = 3,sizey = 1,sizeu = 1,xborders = [0,4],yborders = [0,2],obstacles = [5],obstaclesx = [5],obstaclesy = [5],reward_locations = [5],reward_mags = [5])

# ╔═╡ df146a19-b47b-49eb-993a-7233df3741aa
@with_kw struct params_toy
	ϵ = 0.1
	constant_actions = false
end

# ╔═╡ d30c86b5-61eb-4e79-a485-f0246cae4064
pars_toy = params_toy()

# ╔═╡ f5313ab6-e354-4a9e-837f-a03b09b08d1e
function reachable_states_toy(s_state,a,pars_toy)
	if a == [0,0]
		return [s_state],[1]
	else
		return [s_state,s_state+a],[pars_toy.ϵ,1-pars_toy.ϵ]
	end
end

# ╔═╡ 0beeaba3-e0a7-4975-8e50-0c72ca3df314
function h_iteration_toy(env::environment,pars_toy;tolerance = 1E-2, n_iter = 100,verbose = false)
	value = zeros(env.sizex,env.sizey)
	value_old = deepcopy(value)
	t_stop = n_iter
	f_error = 0
	for t in 1:n_iter
		ferror_max = 0
			for x in 1:env.sizex, y in 1:env.sizey
				s = [x,y]
				actions,_ = adm_actions(s,2,env,pars_toy)
				Z = 0
				for a in actions
					s_primes,probs = reachable_states_toy(s,a,pars_toy)
					expo = 0
					for (j,s_prime) in enumerate(s_primes)
						expo += env.γ*probs[j]*value_old[s_prime[1],s_prime[2]]
					end
					Z += exp(expo/params.α)
				end
				value[x,y] = params.α*log(Z)
				f_error = abs(value[x,y] - value_old[x,y])
				ferror_max = max(ferror_max,f_error)
			end
		if ferror_max < tolerance
			t_stop = t
			break
		end
		if verbose == true 
			println("iteration = ", t, ", max function error = ", ferror_max)
		end
		value_old = deepcopy(value)
	end
	value,t_stop
end

# ╔═╡ 4c79a36c-e679-4fe1-a036-e19f649f4997
val_toy,_ = h_iteration_toy(toy_arena,pars_toy)

# ╔═╡ 0f656395-ffc7-4010-abb9-04b43121bcfb
function optimal_policy_toy(s,optimal_value,env::environment,pars_toy;verbose = false)
	actions,_ = adm_actions(s,2,env,pars_toy)
	policy = zeros(length(actions))
	Z = exp(optimal_value[s[1],s[2]])
	#Only compute policy for available actions
	for (idx,a) in enumerate(actions)
		s_primes,probs = reachable_states_toy(s,a,pars_toy)
		expo = 0
		for (j,s_prime) in enumerate(s_primes)
			expo += env.γ*probs[j]*optimal_value[s_prime[1],s_prime[2]]
		end
		policy[idx] = exp(expo/params.α-optimal_value[s[1],s[2]])
	end
	#adjust for numerical errors in probability
	sum_p = sum(policy)
	if verbose == true
		println("state = ", s, " u = ", u)
		println("policy = ", policy)
		println("sum policy = ", sum(policy))
	end
	policy = policy./sum(policy)
	actions,policy
end

# ╔═╡ 52a823a0-4fb9-4fcf-87d6-beecc2cb1ab2
optimal_policy_toy([2,1],val_toy,toy_arena,pars_toy)

# ╔═╡ b38a0ef8-4889-4f56-a41e-8d5173c5db50
function sample_trajectory_toy(s_0,max_t,opt_value,env,params)
	pos = Any[]
	push!(pos,s_0[1])
	s = deepcopy(s_0)
	for t in 1:max_t
		actions,policy = optimal_policy_toy(s,opt_value,env,params)
		idx = rand(Categorical(policy))
		action = actions[idx]
		s_primes,probs = reachable_states_toy(s,action,params)
		idx_sp = rand(Categorical(probs))
		s_p = s_primes[idx_sp]
		push!(pos,s_p[1])
		s = s_p
	end
	pos
end

# ╔═╡ f66c8ed5-db75-45f0-9962-e63a04caae80
xs = sample_trajectory_toy([3,1],10000,val_toy,toy_arena,pars_toy)

# ╔═╡ 58b6f009-bd69-4bb7-bd44-c395718fae5d
function count_elements(vec,pos)
	c = 0
	for i in 1:length(vec)
		if vec[i] == pos
			c += 1
		end
	end
	c
end

# ╔═╡ 84b2cdf6-1f82-4493-9aa7-97ea74bd9592
begin
	#plot()
	h_agent = histogram2d(Float64.(xs),ones(length(xs)),bins = 3,normalized = :probability,clim = (0.0,1.0),cbar = false,title = "H agent")
	e_mat = zeros(1,3)
	e_mat[1,2] = 1
	e_agent = heatmap(e_mat,cbar = false,title = "E agent")#histogram2d([2],[1],bins = ([0.5,1.5,2.5,3.5],[0.5,1.5]),normalized = :probability,clim = (0.0,1.0))
	s_mat = zeros(1,3)
	s_mat[1,1] = 1
	s_agent = heatmap(s_mat,cbar = false,title = "S agent")
	plot(h_agent,e_agent,s_agent,layout = (3,1),xticks = false,yticks = false,size = (400,400))
	#plot(h_agent)
	#savefig("occupancies_hes.pdf")
end

# ╔═╡ 9eff1101-d2f8-4952-8efe-d8e6ce9bc195
md"# Noisy TV problem"

# ╔═╡ 496c1dbe-1052-45c1-9448-31befea96222
function create_tv_room()
	tv_room = Any[]
	for i in 7:11
		for j in 1:5
			push!(tv_room,[i,j])
		end
	end
	tv_room
end

# ╔═╡ 586a8b9b-1a02-41a0-93c0-85e69f56da90
function reachable_states_tv(s_state,u_state,action,pars,env)
	tv_room = create_tv_room()
	if s_state in tv_room
		actions,_ = adm_actions(s_state,u_state,env)
		states = Any[]
		probs = Any[]
		for a in actions
			push!(states,s_state + a)
			if a == action
				push!(probs,1-pars.η + pars.η/length(actions))
			else
				push!(probs,pars.η/length(actions))
			end
		end
		return states,probs
	else
		return [s_state + action],[1.0]
	end
end

# ╔═╡ 6ad087b7-c848-498b-beba-fbd5c4b0c4c4
function h_iteration_tv(env::environment,params;tolerance = 1E-2, n_iter = 100,verbose = false)
	value = zeros(env.sizex,env.sizey,env.sizeu)
	value_old = deepcopy(value)
	t_stop = n_iter
	f_error = 0
	for t in 1:n_iter
		ferror_max = 0
		Threads.@threads for u in 1:env.sizeu
			for x in 1:env.sizex, y in 1:env.sizey
				s = [x,y]
				ids = findall(i -> i == s,env.obstacles)
				#Only update value for non-obstacle states
				if length(ids) == 0
					actions,_ = adm_actions(s,u,env)
					Z = 0
					expos = zeros(length(actions))
					for (i,a) in enumerate(actions)
						s_primes,probs = reachable_states_tv(s,u,a,params,env)
						
						u_prime = transition_u(s,u,a,env)
						for (j,s_prime) in enumerate(s_primes)
							expos[i] += -params.β*probs[j]*log(probs[j]) + params.γ*probs[j]*value_old[s_prime[1],s_prime[2],u_prime]
						end
						
					end
					#Log sum exp trick
					c = maximum(expos)/params.α
					Z = sum(exp.(expos./params.α .- c))
					value[x,y,u] = params.α*(c + log(Z))
					f_error = abs(value[x,y,u] - value_old[x,y,u])
					ferror_max = max(ferror_max,f_error)
				else
					value[x,y,u] = 0
				end
			end
		end
		if ferror_max < tolerance
			t_stop = t
			break
		end
		if verbose == true 
			println("iteration = ", t, ", max function error = ", ferror_max)
		end
		value_old = deepcopy(value)
	end
	value,t_stop
end

# ╔═╡ 867527c2-e09e-4486-ba06-99a83fda13c6
reward_locations_tv = [[1,1],[size_x,size_y],[size_x,1],[1,size_y]]

# ╔═╡ 5bebc6e9-1bde-40f4-bac4-2f5ec05ab548
begin
	capacity_tv = 100
	food_gain_tv = capacity_tv	
	reward_mags_tv = [1,1,1,1].*food_gain_tv
	env_tv = initialize_fourrooms(size_x,size_y,capacity_tv,reward_locations_tv,reward_mags_tv)
	#one small room
	#obstacles = [[1,4],[2,4],[3,4],[4,3],[4,2],[4,1]]
	#Rewards for the walled environment
	#reward_locations = [[1,6],[3,2],[4,2],[5,2],[6,2]]
	#reward_mags = [30,-40,-30,-20,-10]
end

# ╔═╡ d2df50cb-d163-4c49-b940-35179b7148a5
pars_tv = paras(β = 0.3,γ = 0.999,η = 1,constant_actions = false)

# ╔═╡ ffd1a342-c46e-4016-a24b-fc98e4498890
begin
	# h_tv,_ = h_iteration_tv(env_tv,pars_tv,tolerance = 0.01,verbose = true,n_iter = 100);
	h_tv = readdlm("values_tv/h_value_beta_$(pars_tv.β)_g_$(pars_tv.γ)_eta_$(pars_tv.η)_cap_$(capacity_tv).dat")
	h_tv = reshape(h_tv,env_tv.sizex,env_tv.sizey,env_tv.sizeu)
end;

# ╔═╡ 3fb25daf-c9af-4746-b8a6-91bc01e4d12b
begin
	capacity_local = 100
	food_gain_local = capacity_local
	reward_mags_local = [1,1,1,1].*food_gain_local
	env_local = initialize_fourrooms(size_x,size_y,capacity_local,reward_locations_tv,reward_mags_local)
end

# ╔═╡ 9f11779e-c23e-41d3-a51d-eb88db85c7fd
# #Write value function for several betas and capacities
# begin
# 	βs_to_compute_value = collect(0.0:0.050:0.3)#collect(0.0:0.1:1.0)
# 	#γs_to_compute_value = [0.95,0.97,0.99]
# 	ps_to_compute_value = [0.99,0.995]#[10,50,100] #[0.3,0.7]#[0.1,0.5,0.9]
# 	#for γ in γs_to_compute_value
# 	# capacity_local = 30
# 	# food_gain_local = capacity_local
# 	# reward_mags_local = [1,1,1,1].*food_gain_local
# 	# env_local = initialize_fourrooms(size_x,size_y,capacity_local,reward_locations_tv,reward_mags_local)
# 	for p in ps_to_compute_value
# 		for β in βs_to_compute_value
# 			pars_t = paras(β = β,γ = p,η = 1)		
# 			h_vtv,_ = h_iteration_tv(env_local,pars_t,tolerance = 0.01,verbose = false,n_iter = 1000);
# 			writedlm("values_tv/h_value_beta_$(β)_g_$(pars_t.γ)_eta_$(pars_t.η)_cap_$(capacity_local).dat",h_vtv)
# 		end
# 	end
# end

# ╔═╡ a134c7d5-15e6-4a22-bc34-56a3155b88dd
function optimal_policy_tv(s,u,optimal_value,env::environment,params;verbose = false)
	actions,_ = adm_actions(s,u,env,params)
	policy = zeros(length(actions))
	#Z = exp(optimal_value[s[1],s[2],u])
	#Only compute policy for available actions
	for (idx,a) in enumerate(actions)
		s_primes,probs = reachable_states_tv(s,u,a,params,env)
		expo = 0
		for (j,s_prime) in enumerate(s_primes)
			u_prime = transition_u(s,u,a,env)
			expo += -params.β*probs[j]*log(probs[j]) + params.γ*probs[j]*optimal_value[s_prime[1],s_prime[2],u_prime]
			#@show expo
		end
		policy[idx] = exp(expo/params.α-optimal_value[s[1],s[2],u])
	end
	#adjust for numerical errors in probability
	sum_p = sum(policy)
	if verbose == true
		println("state = ", s, " u = ", u)
		println("policy = ", policy)
		println("sum policy = ", sum(policy))
	end
	policy = policy./sum(policy)
	actions,policy
end

# ╔═╡ f2275b68-335a-4383-9ad9-b2e47286f008
begin
	p_tv = heatmap(transpose(h_tv[:,:,5]),clim=(0,100))
	for i in 1:length(env_tv.reward_mags)
		if env_tv.reward_mags[i]  > 0
			col = "green"
		else
			col = "gray"
		end
		scatter!(p_tv,[env_tv.reward_locations[i][1]],[env_tv.reward_locations[i][2]], color = col, markershape = :diamond, markersize = min(abs(env_tv.reward_mags[i]),10))
	end
	plot!(p_tv,size = (330,300)) 
	#plot_optimal_policy(p1,u,h_value,env_c)
	#p2 = heatmap(val_stoch[:,:,u], title = "value function random walk")
	plot(p_tv,legend = false,axis = false,ticks =false,margin = 5Plots.mm)
	#savefig("value_noisy_tv.pdf")
end

# ╔═╡ 53e6d8b6-fe76-4b1a-b6e4-8c55fc58db7d
function sample_trajectory_tv(s_0,u_0,opt_value,max_t,env::environment,params,occupancies = false)
	xpositions = Any[]
	ypositions = Any[]
	u_states = Any[]
	all_x = Any[]
	all_y = Any[]
	dead_times = Any[]
	push!(xpositions,[s_0[1]])
	push!(ypositions,[s_0[2]])
	push!(all_x,s_0[1])
	push!(all_y,s_0[2])
	push!(u_states,u_0)

	t_max = max_t
	
	s = deepcopy(s_0)
	u = deepcopy(u_0)

	for t in 1:max_t
		actions,policy = optimal_policy_tv(s,u,opt_value,env,params)
		#@show policy
		idx = rand(Categorical(policy))
		action = actions[idx]
		s_primes,probs = reachable_states_tv(s,u,action,params,env)
		idx_sp = rand(Categorical(Float64.(probs)))
		s_p = s_primes[idx_sp]
		u_p = transition_u(s,u,action,env)
		s = s_p
		u = u_p
		push!(xpositions,[s[1]])
		push!(ypositions,[s[2]])
		push!(all_x,s[1])
		push!(all_y,s[2])
		push!(u_states,u)
		if u == 1
			break
			#u = env.sizeu
			#push!(dead_times,t+1)
		end
	end
	xpositions,ypositions,u_states,all_x,all_y,dead_times
end

# ╔═╡ b63f2611-0bca-4e68-bd1d-dc9cef75d9f1
begin
	βs = collect(0.0:0.050:0.3)
	γs = [0.95,0.99,0.995,0.999]
	ηs = [0.1,0.3,0.5,0.7]
	cs = [10,50,100]
	ps = γs
	max_time_room = 1000
	num_episodes_room = 100
	time_in_room = zeros(length(ps),length(βs),num_episodes_room)
	surv_times_tv = zeros(length(ps),length(βs),num_episodes_room)
	# #for (i,γ) in enumerate(γs)
	# for (i,p) in enumerate(ps)
	# 	for (j,β) in enumerate(βs)
	# 		#@show β
	# 		par = paras(β = β,γ = p,η = 1)
	# 		h_val = readdlm("values_tv/h_value_beta_$(β)_g_$(par.γ)_eta_$(par.η)_cap_$(capacity_local).dat")
	# 		h_val = reshape(h_val,env_local.sizex,env_local.sizey,env_local.sizeu)
	# 		Threads.@threads for k in 1:num_episodes_room
	# 			#println("j = ", j)
	# 			_,_,_,xs_tv,ys_tv,dead_times_tv = sample_trajectory_tv([3,3],env_local.sizeu,h_val,max_time_room,env_local,par);
	
	# 			time_in_room[i,j,k] = length([[xs_tv[l],ys_tv[l]] for l in 1:length(xs_tv) if xs_tv[l] > 6 && ys_tv[l] < 6])/length(xs_tv)
	# 			#time_stochastic_a[i,j] = length([a for a in x_half_a if a > 0])/length(xposcar_half_a)
	# 			#survival_timesq[j] = length(xposcar_q)
	# 			#if length(xposcar_q) == max_time
	# 			surv_times_tv[i,j,k] = length(xs_tv)
	# 			#survival_pcts_a[i,j] = length(xposcar_half_a)
	# 			#end
	# 		end
	# 	end
	# end
	# writedlm("tv_room/time_room_betas_$(βs)_p_$(ps)_time_$(max_time_room).dat",time_in_room)

	# writedlm("tv_room/survival_room_betas_$(βs)_p_$(ps)_time_$(max_time_room).dat",surv_times_tv)

	# #Otherwise read from file
	time_in_room = readdlm("tv_room/time_room_betas_$(βs)_p_$(ps)_time_$(max_time_room).dat")
	surv_times_tv = readdlm("tv_room/survival_room_betas_$(βs)_p_$(ps)_time_$(max_time_room).dat")
	time_in_room = reshape(time_in_room,length(ps),length(βs),num_episodes_room)
	surv_times_tv = reshape(surv_times_tv,length(ps),length(βs),num_episodes_room)
end;

# ╔═╡ 20cba048-e6c8-42d3-ad15-8091a8fd9bfc
begin
	surv_means = plot(xticks = collect(0.0:0.1:0.3),xlabel = "\$\\beta\$",ylabel = "Survived time steps",yticks = [0,500,1000],ylim = (0,1100))
	colors = ["#D3D3D3","#C0C0C0","#696969","#000000"]
	for (i,p) in enumerate(ps[2:4])
		i = i+1
		plot!(surv_means,βs,mean(surv_times_tv[i,:,:],dims =2),label = "\$\\gamma\$ = $(p)",linewidth = 2.5,yerror = std(surv_times_tv[i,:,:],dims = 2)./sqrt(length(surv_times_tv[i,1,:])),markerstrokewidth = 1,lc = colors[i],msc = :auto)
		plot!(surv_means, βs, mean(surv_times_tv[i,:,:],dims =2),label = false,color = colors[i],markersize = 3,st = :scatter,msc = :auto)
	end
	plot(surv_means, grid = false, legend_position = :bottomleft,margin = 2Plots.mm,size = (450,300),bg_legend = :transparent,fg_legend = :transparent,fgguide = :black)
	#savefig("tv_room/h_betas_$(βs)_p_$(ps)_survival_time_$(max_time_room).pdf")
end

# ╔═╡ 9b45f045-77d6-43c1-a7a0-5d4d5c7af480
begin
	times_means = plot(ylim=(-0.05,1.1),xticks = collect(0.0:0.1:1.0),yticks = collect(0.0:0.25:1.0),xlabel = "\$\\beta\$",ylabel = "Time fraction in noisy room")
	for (i,p) in enumerate(ps[2:4])
		i = i+1
		plot!(times_means,βs,mean(time_in_room[i,:,:],dims =2),label = "\$\\gamma\$ = $(p)",linewidth = 2,yerror = std(time_in_room[i,:,:],dims = 2)./sqrt(length(time_in_room[i,1,:])),markerstrokewidth = 1,markersize = 4,lc = colors[i],msc = :auto)
		plot!(times_means, βs, mean(time_in_room[i,:,:],dims =2),color = colors[i],st = :scatter,markersize = 3,msc = :auto,label = false)
		#plot!(times_means_half_β, βs, mean(time_stochastic_β[i,:,:],dims =2),label = "η = $(η)",linewidth = 2.5,yerror = std(time_stochastic_β[i,:,:],dims = 2)./sqrt(length(time_stochastic_β[i,1,:])))
	end


	plot(times_means, grid = false, legend_position = :topleft,margin = 2Plots.mm,size = (500,350),bg_legend = :transparent,fg_legend = :transparent,fgguide = :black)
	#savefig("tv_room/h_betas_$(βs)_p_$(ps)_timeinroom.pdf")
end

# ╔═╡ e7578dbf-ac6c-414c-9e08-1ed9636177f7
md"# Random walker"

# ╔═╡ 6b3fb79f-be03-4d90-9527-83e868cdaddd
function sample_trajectory_random(s_0,u_0,max_t,env::environment,pars,occupancies = true)
	xpositions = Any[]
	ypositions = Any[]
	u_states = Any[]
	all_x = Any[]
	all_y = Any[]
	push!(xpositions,[s_0[1]])
	push!(ypositions,[s_0[2]])
	push!(all_x,s_0[1])
	push!(all_y,s_0[2])
	push!(u_states,u_0)
	unvisited_s_states = Any[]
	unvisited_u_states = collect(1:env.sizeu)
	n_arena_states = env.sizex*env.sizey - length(env.obstacles)
	for x in 1:env.sizex
		for y in 1:env.sizey
			if ([x,y] in env.obstacles) == false
				push!(unvisited_s_states,[x,y])
			end
		end
	end
	s = deepcopy(s_0)
	u = deepcopy(u_0)
	id_s = findfirst(i -> i == [s[1],s[2]],unvisited_s_states)
	id_u = findfirst(i -> i == u,unvisited_u_states)
	deleteat!(unvisited_s_states,id_s)
	deleteat!(unvisited_u_states,id_u)
	for t in 1:max_t
		actions_at_s,_ = adm_actions(s,u,env,pars)
		action = rand(actions_at_s)
		#action = rand(actions)
		s_p = transition_s(s,action,env)
		u_p = transition_u(s,u,action,env)
		s = s_p[1]
		u = u_p
		#If agent dies, terminate episode
		if u == 1
			return xpositions,ypositions,u_states,all_x,all_y,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,t
		end
		id_s = findfirst(i -> i == [s[1],s[2]],unvisited_s_states)
		id_u = findfirst(i -> i == u,unvisited_u_states)
		if id_s != nothing
			deleteat!(unvisited_s_states,id_s)
		end
		if id_u != nothing
			deleteat!(unvisited_u_states,id_u)
		end
		push!(xpositions,[s[1]])
		push!(ypositions,[s[2]])
		push!(all_x,s[1])
		push!(all_y,s[2])
		push!(u_states,u)		
		if occupancies == true
			if length(unvisited_s_states) == 0
				return xpositions,ypositions,u_states,all_x,all_y,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,t
			end
		end
	end
	xpositions,ypositions,u_states,all_x,all_y,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,max_t
end

# ╔═╡ 5b1ba4f6-37a9-4e61-b6cc-3d495aa67c9d
md"# Comparison between agents"

# ╔═╡ a0dc952c-e733-41de-8d6e-458d66c3769a
md"## Max ent RL"

# ╔═╡ 37d7baa7-b22b-4919-8910-43292287b230
function write_maxent_value_temps(temps)
	for t in temps
		par = paras(α = t)
		maxent_value, t_maxent = maxent_iteration(env1,par,tolerance = 0.1,n_iter = 100)
		writedlm("./maxent_values_temps/val_temp_$(t).dat",maxent_value)
	end
end

# ╔═╡ cb8b2b44-13ee-4c12-833a-26f6d806724b
params

# ╔═╡ a14c2f1f-ad38-45e5-93a9-d6f19641687b
function write_H_value_capacities(capacities,params)
	for c in capacities
		mags = [1,1,1,1].*c
		env_c = initialize_fourrooms(size_x,size_y,c,reward_locations,mags)
		#par = paras(α = t)
		h_value, t_maxent = h_iteration(env_c,params,tolerance = 0.1,n_iter = 100)
		writedlm("./h_values_capacities/val_cap_$(c).dat",h_value)
	end
end

# ╔═╡ 36276897-e845-46b4-8883-a8255d5ecc9c
capacities = [2,3,5,10,20,30]

# ╔═╡ 87a66b34-e7e3-4d3b-9da1-089ad296c643
#write_H_value_capacities(capacities,params)

# ╔═╡ dd9005fe-8777-4f71-aa7e-8de959dcbc52
temps = [0.05,0.1,0.3,0.5,1.0,5.0]

# ╔═╡ ffaa2c35-f6c9-45de-b67c-861dd1aec914
#write_maxent_value_temps([0.3])

# ╔═╡ 91e0e0ca-d9df-44d7-81e6-00b343ad9bf0
md"## Many gains"

# ╔═╡ 1bb9994a-ed89-4e08-921e-39d46fc45e0a
@with_kw struct parameters
	size_x = 11
	size_y = 11
	capacity = 100
	reward_locations = [[1,1],[size_x,size_y],[size_x,1],[1,size_y]]
end

# ╔═╡ beb1a211-f262-49d8-a3c4-73a0cf727921
begin
	energy_gains = [2,3,4,5,6,7,8,9,10]
	ϵs = collect(0.0:0.05:0.5)
end

# ╔═╡ 5d0ad59b-366e-4660-9350-92d34d616f16
pars = parameters()

# ╔═╡ c7d270aa-9c5c-461b-ac6b-2b9287a2d461
function write_valuefunctions_to_files(energy_gains,pars,ϵ)
	tol = 1E-2
	n_iter = 10000
	for gain in energy_gains
		println("gain = ", gain)
		reward_mags = [1,1,1,1].*gain
		env_iteration = initialize_fourrooms(pars.size_x,pars.size_y,pars.capacity,pars.reward_locations,reward_mags)
		h_value,t_stop = h_iteration(env_iteration,tol,n_iter)
		q_value,t_stop_q = q_iteration(env_iteration,ϵ,tol,n_iter,true)
		writedlm("values/h_value_gain_$(gain).dat",h_value)
		writedlm("values/q_value_gain_$(gain).dat",q_value)
	end
end

# ╔═╡ 5cc21644-35f9-404c-8f68-c1591a19226e
function write_valuefunctions_to_files_eps(energy_gains,ϵs,pars)
	tol = 1E-2
	n_iter = 10000
	for gain in energy_gains
		println("gain = ", gain)
		reward_mags = [1,1,1,1].*gain
		env_iteration = initialize_fourrooms(pars.size_x,pars.size_y,pars.capacity,pars.reward_locations,reward_mags)
		h_value,t_stop = h_iteration(env_iteration,tol,n_iter)
		writedlm("values/h_value_gain_$(gain).dat",h_value)
		for ϵ in ϵs
			q_value,t_stop_q = q_iteration(env_iteration,ϵ,tol,n_iter)
			writedlm("values/q_value_gain_$(gain)_eps_$(ϵ).dat",q_value)
		end
	end
end

# ╔═╡ aea98aa9-46a5-47d7-b78f-3b784bcb8668
begin
	#Only run if wanting to modify value functions
	#write_valuefunctions_to_files_eps([20,30,50,75,100],[0.45],pars)
end

# ╔═╡ 8e1241e7-6a23-46cb-8b8b-a6c828fd6f17
begin
	#Only run if wanting to modify value functions
	#write_valuefunctions_to_files_eps([3],[1.0],pars)
end

# ╔═╡ 1767238d-9301-4e84-8b9c-54b8369065d9
function epsilon_gains_occupancies(s_0,u_0,energy_gains,ϵs,n_episodes,t_episode,pars)
	h_life_expectancies = zeros(length(energy_gains),n_episodes)
	q_life_expectancies = zeros(length(energy_gains),length(ϵs),n_episodes)
	h_s_occupancies = zeros(length(energy_gains),n_episodes)
	h_u_occupancies = zeros(length(energy_gains),n_episodes)
	q_s_occupancies = zeros(length(energy_gains),length(ϵs),n_episodes)
	q_u_occupancies = zeros(length(energy_gains),length(ϵs),n_episodes)
	Threads.@threads for i in 1:length(energy_gains)
		println("gain = ", energy_gains[i])
		h_val = reshape(readdlm("values/h_value_gain_$(energy_gains[i]).dat"),pars.size_x,pars.size_y,pars.capacity)
		reward_mags = [1,1,1,1].*energy_gains[i]
		env_iteration = initialize_fourrooms(pars.size_x,pars.size_y,pars.capacity,pars.reward_locations,reward_mags)
		for j in 1:n_episodes
			println("episode = ", j)
	_,_,_,_,_,_,h_s_occupancies[i,j],h_u_occupancies[i,j],h_life_expectancies[i,j] = sample_trajectory(s_0,u_0,h_val,t_episode,env_iteration)
			for (k,ϵ) in enumerate(ϵs)
			q_val = reshape(readdlm("values/q_value_gain_$(energy_gains[i])_eps_$(ϵ).dat"),pars.size_x,pars.size_y,pars.capacity)
			_,_,_,_,_,q_s_occupancies[i,k,j],q_u_occupancies[i,k,j],q_life_expectancies[i,k,j] = sample_trajectory_q(s_0,u_0,q_val,ϵ,t_episode,env_iteration)
			end
		end
	end
	h_s_occupancies,h_u_occupancies,h_life_expectancies,q_s_occupancies,q_u_occupancies,q_life_expectancies
end

# ╔═╡ f79a8a77-02f5-4677-a34f-e6a23be4d5ad
begin
	#This computes occupancies running n_episodes, each with a max duration of t_episode, for both Q and H agent. It takes up to 8 hours for t_episode = 1E6 and 50 episodes.
	t_ep = 50000
	n_ep = 50
	e_gains = [2,3,5,10,20,30,50,75,100]
	#epsilons for final paper
	ϵs_new = [0.0,0.45,0.5,1.0]
	#epsilons for final final
	#ϵs_new = [0.0,0.3,0.35,0.4,1.0]
	#epsilons for final2
	#ϵs_new = [0.0,0.1,0.3,0.5,0.6,0.7,0.9,1.0]
# h_s_occ,h_u_occ,surv_times_h,q_s_occ,q_u_occ,surv_times_q = epsilon_gains_occupancies([3,3],Int(pars.capacity),e_gains,ϵs_new,n_ep,t_ep,pars)
end

# ╔═╡ ebe880f4-7001-4118-b523-037af729d95f
# begin
# 	writedlm("occupations_final_paper/h_s_occ.dat",h_s_occ)
# 	writedlm("occupations_final_paper/h_u_occ.dat",h_u_occ)
# 	writedlm("occupations_final_paper/h_survival.dat",surv_times_h)
# 	writedlm("occupations_final_paper/q_s_occ.dat",q_s_occ)
# 	writedlm("occupations_final_paper/q_u_occ.dat",q_u_occ)
# 	writedlm("occupations_final_paper/q_survival.dat",surv_times_q)
# end

# ╔═╡ 8a6d0254-211f-45dd-8798-8408cb9ccc55
ϵs_read = ϵs_new

# ╔═╡ 96a54510-789e-455c-bbab-390f78e909cd
begin
	#ϵs_read = ϵs_new
	h_s_occ = reshape(readdlm("occupations_final_paper/h_s_occ.dat"),length(e_gains),n_ep)
	h_u_occ=reshape(readdlm("occupations_final_paper/h_u_occ.dat"),length(e_gains),n_ep)
	surv_times_h=reshape(readdlm("occupations_final_paper/h_survival.dat"),length(e_gains),n_ep)
	q_s_occ=reshape(readdlm("occupations_final_paper/q_s_occ.dat"),length(e_gains),length(ϵs_read),n_ep)
	q_u_occ=reshape(readdlm("occupations_final_paper/q_u_occ.dat"),length(e_gains),length(ϵs_read),n_ep)
	surv_times_q=reshape(readdlm("occupations_final_paper/q_survival.dat"),length(e_gains),length(ϵs_read),n_ep)
end;

# ╔═╡ d2f49d6b-2acc-4fef-8e24-9acdd7591977
cols = palette(:Dark2_8)

# ╔═╡ 6dcaa4f7-a539-412f-8ee0-f1f9dfba831f
indices_to_plot = [2,4]

# ╔═╡ abb61da7-3711-4fa6-a021-2bb916411019
begin
	colors_a = ["blue","orange","green"]
	plot(ylabel = "Fraction occupied arena",xlabel = "Food gain")#, xlim = (2,14),xticks = collect(2:10))
	plot!(xscale = :log10,legend_position = (0.52,0.27),bg_color_legend = :transparent, fg_color_legend = :transparent,grid = false,xminorticks = 9, size = (450,330), margin = 2Plots.mm,ylim= (-0.05,1.05))
	
	occ_h = 1 .-((1 .-(h_s_occ)).*(env1.sizex*env1.sizey-length(env1.obstacles)))./(env1.sizex*env1.sizey-33)
	
	plot!(e_gains,mean(occ_h,dims=2),yerror = std(occ_h,dims = 2)./(sqrt(n_ep)), label = "H agent",lw = 2.,lc = colors_a[1],msc = :auto)
	plot!(e_gains,mean(occ_h,dims=2),st = :scatter,c = colors_a[1], label = false,ms= 3,msc = :auto)
	
	plot!(e_gains,mean(q_s_occ[:,2,:],dims=2),st = :line, yerror = std(q_s_occ[:,2,:],dims = 2)./(sqrt(n_ep)), label = "R agent, ϵ = 0.45",lw = 2.,lc = colors_a[2],msc = :auto,mc = colors_a[2])
	plot!(e_gains,mean(q_s_occ[:,2,:],dims=2),st = :scatter,color = colors_a[2], label = false,ms = 3.,msc = :auto)
	
	plot!(e_gains,mean(q_s_occ[:,4,:],dims=2),yerror = std(q_s_occ[:,4,:],dims = 2)./(sqrt(n_ep)), label = "Random walker",lw = 2.,lc = colors_a[3],msc = :auto )
	plot!(e_gains,mean(q_s_occ[:,4,:],dims=2), st = :scatter,mc = colors_a[3],msc = :auto, label = false,ms = 3)
	#hline!([0], label = false, color = :black,ls = :dash)

	#savefig("occupations_rooms_paper.pdf")
	#plot!(surv_times_h[1,:],st = :histogram)
end

# ╔═╡ 94430097-f64f-4d27-aa0f-24aec2042d5d
begin
	plot(ylabel = "Survived time steps", xlabel = "Food gain")#, xlim = (2,14),xticks = collect(2:10))
	plot!(e_gains,mean(surv_times_h,dims = 2),label = "H agent",lw = 4, color = colors_a[1])
	# for i in indices_to_plot
	# 	plot!(e_gains,mean(surv_times_q[:,i,:],dims=2), yerror = std(surv_times_q[:,i,:],dims = 2)./sqrt(n_ep),c = cols[i], label = "ϵ = $(ϵs_read[i])",lw = 2)
	# end
	plot!(e_gains,mean(surv_times_q[:,2,:],dims=2), yerror = std(surv_times_q[:,2,:],dims = 2)./sqrt(n_ep), label = "R agent,ϵ = $(ϵs_read[2])",lw = 2,color = colors_a[2])
	plot!(e_gains,mean(surv_times_q[:,3,:],dims=2), yerror = std(surv_times_q[:,3,:],dims = 2)./sqrt(n_ep), label = "R agent,ϵ = $(ϵs_read[3])",lw = 2, color = "brown")
	plot!(e_gains,mean(surv_times_q[:,4,:],dims=2), yerror = std(surv_times_q[:,4,:],dims = 2)./sqrt(n_ep), label = "Random walker",lw = 2,color = colors_a[3])
	plot!(scale = :log,legend_position = :right, grid = false, fg_color_legend = :white,margin = 2Plots.mm, size = (450,300))
	#savefig("survival_rooms_paper.pdf")
	#plot!(surv_times_h[1,:],st = :histogram)
end

# ╔═╡ 1c69c08b-aae5-453e-98ef-df35c7b4db50
function occupancies_qh(energy_gains,n_episodes,t_episode,pars)
	s_0 = [3,3]
	u_0 = Int(pars.capacity/2)
	h_s_occupancies = zeros(length(energy_gains),n_episodes)
	h_u_occupancies = zeros(length(energy_gains),n_episodes)
	q_s_occupancies = zeros(length(energy_gains),n_episodes)
	q_u_occupancies = zeros(length(energy_gains),n_episodes)
	times_h = zeros(length(energy_gains),n_episodes)
	times_q = zeros(length(energy_gains),n_episodes)
	for (i,gain) in enumerate(energy_gains)
		println("gain = ", gain)
		h_val = reshape(readdlm("values/h_value_gain_$(gain).dat"),pars.size_x,pars.size_y,pars.capacity)
		q_val = reshape(readdlm("values/q_value_gain_$(gain).dat"),pars.size_x,pars.size_y,pars.capacity)
		reward_mags = [1,1,1,1].*gain
		env_iteration = initialize_fourrooms(pars.size_x,pars.size_y,pars.capacity,pars.reward_locations,reward_mags)
		for j in 1:n_episodes
		println("episode = ", j)
			_,_,h_us,h_allx,h_ally,_,h_s_occupancies[i,j],h_u_occupancies[i,j],times_h[i,j] = sample_trajectory(s_0,u_0,h_val,t_episode,env_iteration)
			_,_,q_us,q_allx,q_ally,q_s_occupancies[i,j],q_u_occupancies[i,j],times_q[i,j] = sample_trajectory_q(s_0,u_0,q_val,t_episode,env_iteration)
		end
	end
	h_s_occupancies, h_u_occupancies, q_s_occupancies, q_u_occupancies, times_h, times_q
end

# ╔═╡ 0a9dc717-dbcb-4b27-8c76-cb8fbfdbec96
begin
	n_episodes = 50
	t_episode = 1000000
end

# ╔═╡ 1f3be0d8-d296-4eb0-a951-bd862914ae92
md"## One long episode"

# ╔═╡ bb45134a-88b1-40d2-a486-c7afe8ac744e
begin
	q_value_50 = q_value #readdlm("q_value_u_50.dat")
	h_value_50 = h_value #readdlm("h_value_u_50.dat")
	#q_value_200 = readdlm("q_value_u_200.dat")
	#h_value_200 = readdlm("h_value_u_200.dat")
	#q_value_50 = reshape(q_value_50,env1.sizex,env1.sizey,env1.sizeu)
	#h_value_50 = reshape(h_value_50,env1.sizex,env1.sizey,env1.sizeu)
	#q_value_200 = reshape(q_value_200,env1.sizex,env1.sizey,env2.sizeu)
	#h_value_200 = reshape(h_value_200,env1.sizex,env1.sizey,env2.sizeu)
end;

# ╔═╡ f1d5ee65-10f5-424a-b018-2aff1a5d7ff8
begin
		#Initial condition
		s_0 = [3,3] #[1,env.sizey]
		u_0 = 100#Int(env1.sizeu)
end

# ╔═╡ bd16a66c-9c2f-449c-a792-1073c54e990b
begin
	draw_environment([3],[3],[u_0],env1,params)
	#savefig("arena.pdf")
end

# ╔═╡ 78a5caf6-eced-4783-b950-26563f632be2
begin
	@bind max_t Select([50000 => "short episode", 1E5 => "long episode"], [1000])
end

# ╔═╡ 4f5d1b29-90a2-4b79-9ce0-30764e6d3350
begin
	_,_,_,q_xpos_fa,q_ypos_fa,_ = sample_trajectory_q([3,3],50,q_value_fa,0.45,max_t,env1,params_fa,false)
end

# ╔═╡ e1c59a22-fdd8-4edd-86d5-9efd9aca69a5
states_KL_fa,_,_,_ = create_episode_KL(s_0,50,KL_value_fa,max_t,env1,params_fa,fd = true)

# ╔═╡ 55fdc810-fb0c-4d63-8010-f65200bceb5d
states_KL_va,_,_,_ = create_episode_KL(s_0,50,KL_value_va,max_t,env1,params)

# ╔═╡ e3ff07ca-6c46-4be9-bf78-3c09bed248f6
length(states_KL_va),length(states_KL_fa)

# ╔═╡ d6d26401-22eb-4e8d-8b2a-13223167a949
max_t

# ╔═╡ 0a55db1e-4dc7-4195-aa2f-564d770afa8c
states_AIF,us_AIF,actions_AIF,values_AIF = create_episode_AIF(s_0,50,v_AIF,max_t,env_AIF,p_AIF)

# ╔═╡ f0cb716c-620b-4632-9022-f7b9880ac98e
length(states_AIF),max_t_anim

# ╔═╡ b17456b8-3d6a-4c31-95f2-036c5fd90ea3
s_0,u_0,max_t

# ╔═╡ 752633e4-ed90-4cab-819f-af03bcf890c7
maxent_xpos,maxent_ypos,maxent_us,maxent_allx,maxent_ally,maxent_urgency = sample_trajectory(s_0,u_0,maxent_value,max_t,env1,params)

# ╔═╡ eef8a58c-4163-49a2-bbdf-48f9f2782a08
begin
	rate_c = zeros(length(capacities))
	for (i,c) in enumerate(capacities)
		#par = paras(α = t)
		mags = [1,1,1,1].*c
		env_c2 = initialize_fourrooms(size_x,size_y,c,reward_locations,mags)
		value = reshape(readdlm("./h_values_capacities/val_cap_$(c).dat"),env_c2.sizex,env_c2.sizey,env_c2.sizeu)
		u_c = c
		_,_,h_us,_,_,_,_,_,t_max,dead_times = sample_trajectory([1,1],u_c,value,max_t,env_c2,params)
		if any(h_us .== 1) == true
			println(c, "dead")
		end
		@show c,length(dead_times)
		rate_c[i] = (length(findall(a -> a >=0, diff(h_us)))-length(dead_times))/(max_t-length(dead_times))
	end
end

# ╔═╡ dc8dd49b-cb51-45cc-8cd7-ffa497b3f153
begin
	plot(capacities,rate_c,label = "H agent", xscale = :log10, markershape = :circle,xlabel = "Capacity", ylim = (0.0,1.0),yticks = collect(0.0:0.2:1.0), xticks = [10^0.5,10,10^1.5],ylabel = "Eating rate",margin = 4Plots.mm, grid = false, legendforegroundcolor = :white, color = "blue",ms = 2.5,size = (450,300))
	#savefig("capacities.pdf")
end

# ╔═╡ fd4463de-56de-4707-8a17-f5be5a0cd33e
begin
	rate = zeros(length(temps))
	for (i,t) in enumerate(temps)
		par = paras(α = t)
		value = reshape(readdlm("./maxent_values_temps/val_temp_$(t).dat"),env1.sizex,env1.sizey,env1.sizeu)
		_,_,maxent_us,_,_,_,_,_,t_max,dead_times = sample_trajectory(s_0,u_0,value,max_t,env1,par)
		if any(maxent_us .== 1) == true
			println("dead")
		end
		@show t,length(dead_times)
		rate[i] = (length(findall(a -> a >=0, diff(maxent_us)))-length(dead_times))/(max_t-length(dead_times))
	end
end

# ╔═╡ 676487d4-839b-4641-ad77-af3f1610787f
begin
	plot(temps,rate,markershape = :circle, color = cols[4],xscale = :log10,xlabel = "Temperature", yticks = collect(0.0:0.2:1.0), ylabel = "Eating rate",label = "Max Ent RL",margin = 4Plots.mm, grid = false, legendforegroundcolor = :white,ms = 2.5, size = (450,300))
	#savefig("maxent_temperature.pdf")
end

# ╔═╡ 4a868ec2-b636-4d5d-a248-0a4e0cca3668
begin
	# h_xpos_50,h_ypos_50,h_us_50,h_allx_50,h_ally_50,h_urgency_50,h_visited_s,h_visited_u,h_time,d_times_50 = sample_trajectory(s_0,u_0,h_value_50,max_t,env1,params,false)
	#h_xpos_200,h_ypos_200,h_us_200,h_allx_200,h_ally_200,h_urgency_200 = sample_trajectory(s_0,u_0,h_value_200,max_t,env2)
end

# ╔═╡ 6edb5b48-b590-492d-bd3e-6a4f549aae30
begin
	# q_xpos_50,q_ypos_50,q_us_50,q_allx_50,q_ally_50,q_visited_s,q_visited_u,q_time = sample_trajectory_q(s_0,u_0,q_value_50,ϵ,max_t,env1,false)
	#q_xpos_200,q_ypos_200,q_us_200,q_allx_200,q_ally_200 = sample_trajectory_q(s_0,u_0,q_value_200,max_t,env2)
end;

# ╔═╡ 4c51afa2-e294-4922-9cae-d087582d771c
function many_episodes_histogram(s_0,u_0,h_value,q_value,ϵ,max_t,n_episodes,env::environment,params)
	hs_x = Any[]
	hs_y = Any[]
	qs_x = Any[]
	qs_y = Any[]
	rand_x = Any[]
	rand_y = Any[]
	times_h = Any[]
	times_q = Any[]
	times_rand = Any[]
	hs_u = Any[]
	for j in 1:n_episodes
		println("ep = ", j)
		h_xpos,h_ypos,h_us,h_allx,h_ally,h_urgency,h_visited_s,h_visited_u,h_time = sample_trajectory(s_0,u_0,h_value,max_t,env,params,false)
		q_xpos,q_ypos,q_us,q_allx,q_ally,q_visited_s,q_visited_u,q_time = sample_trajectory_q(s_0,u_0,q_value,ϵ,max_t,env,params,false)
		rand_xpos,rand_ypos,rand_us,rand_allx,rand_ally,rand_visited_s,rand_visited_u,rand_time = sample_trajectory_random(s_0,u_0,max_t,env,params,false)
		push!(times_h,h_time)
		push!(times_q,q_time)
		push!(times_rand,rand_time)
		for i in 1:length(h_allx)
			push!(hs_x,h_allx[i])
			push!(hs_y,h_ally[i])
		end
		for i in 1:length(q_allx)
			push!(qs_x,q_allx[i])
			push!(qs_y,q_ally[i])
		end
		for i in 1:length(rand_allx)
			push!(rand_x,rand_allx[i])
			push!(rand_y,rand_ally[i])
		end		
		hs_u = h_us
	end
	hs_x,hs_y,hs_u,qs_x,qs_y,rand_x,rand_y,times_h,times_q,times_rand
end

# ╔═╡ c842a822-462b-4849-86f1-3fc717c492c1
n_episodes_hist = 1

# ╔═╡ 42423e46-2446-4cdd-84c1-fe824e5eb3c4
begin
	import Random
	Random.seed!(11)
	hs_x,hs_y,hs_u,qs_x,qs_y,rand_x,rand_y,times_h,times_q,times_rand = many_episodes_histogram(s_0,u_0,h_value_50,q_value_50,ϵ,max_t,n_episodes_hist,env1,params)
end

# ╔═╡ a0729563-0b6d-4014-b8c7-9eb284a34606
if movie
	Random.seed!(2)
	anim_h = animation(h_xpos_anim,h_ypos_anim,h_us_anim,max_t_anim,env1,color = palette(:default)[1])
end

# ╔═╡ 11b5409c-9db8-4b34-a111-7a62fedd23be
gif(anim_h, fps = 12)#, "hagent.gif")

# ╔═╡ 787bbe73-6052-41e0-bc8c-955e4a884886
if movie_q
	Random.seed!(2)
	q_xpos_anim,q_ypos_anim,q_us_anim,q_allx_anim,q_ally_anim= sample_trajectory_q(s_0,u_0,q_value,ϵ,max_t_anim,env1)
	anim_q = animation(q_xpos_anim,q_ypos_anim,q_us_anim,max_t_anim,env1, title = "R agent, ϵ = $(ϵ)", color = palette(:default)[2])
	#gif(anim_q, "q_agent.gif", fps = 8)
end

# ╔═╡ 139c806d-3f52-4fb9-9fe8-c57259ed1b6f
gif(anim_q, fps = 12, "qagent.gif")

# ╔═╡ a5f43388-9e45-497f-b0f8-7f2987b3102d
anim_both = animation2(h_xpos_anim,h_ypos_anim,h_us_anim,q_xpos_anim,q_ypos_anim,q_us_anim,max_t_anim,env1)

# ╔═╡ 293c129d-dba4-4b04-aa0a-66e1b570aad4
gif(anim_both, fps = 10, "both_agents.gif")

# ╔═╡ 01606177-e88b-453c-ae34-8b3f5755bb82
begin
	Random.seed!(11)
	_,_,_,h_xpos_fa,h_ypos_fa,_= sample_trajectory([3,3],50,h_value_fa,max_t,env1,params_fa)
end

# ╔═╡ 4b5eaa5a-df95-4060-bb99-1e69ccaf8533
h_xpos_fa

# ╔═╡ d66ab0d0-249d-464b-92d7-a0493338b7d0
begin
	Random.seed!(11)
	xpos_tv,ypos_tv,_,xs_tv,ys_tv,dead_times_tv = sample_trajectory_tv([3,3],env_tv.sizeu,h_tv,50000,env_tv,pars_tv);
end

# ╔═╡ 759ae341-ec69-4557-a845-cfcbcedb26dd
begin
	## H agent
	max_tv = 0.05
	p_h_hist_tv = plot(ticks = false)#,ylabel = "\$u_{max} = 50\$")
	#Paint the whole arena 
	heatmap!(p_h_hist_tv,zeros(env_tv.sizex,env_tv.sizey),clim = (0,max_tv))
	#Draw obstacles
	plot!(p_h_hist_tv,env_tv.obstaclesx,env_tv.obstaclesy,bins = (collect(0.5:1:env_tv.sizex + 0.5),collect(0.5:1:env_tv.sizey + 0.5)), st = :histogram2d, color = "black",clim = (0,max_tv))
	#Draw histogram
	histogram2d!(p_h_hist_tv,xs_tv,ys_tv, bins = (collect(0.5:1:env_tv.sizex + 0.5),collect(0.5:1:env_tv.sizey + 0.5)),normalize = :probability,clim = (0,max_tv))
	plot!(p_h_hist_tv)
	plot(p_h_hist_tv,margin = 8Plots.mm, size = (330,300))
	#savefig("noisy_tv_problem_highgamma_beta0.3.pdf")
end

# ╔═╡ d3a2593d-0d55-470c-bff1-4d80714a6f3a
begin
	bins_MOP = (collect(1:env1.sizex),collect(1:env1.sizey),collect(1:env1.sizeu))
	h_MOP = fit(Histogram, (hs_x,hs_y,hs_u), bins_MOP)
	h_n_MOP = normalize(h_MOP,mode =:probability)
	entropy_MOP = entropy(h_n_MOP.weights)
end

# ╔═╡ 188dc342-f0ae-4df1-adf6-bce3da9d5d76
begin
	plot(xlabel = "Horizon",ylabel = "Entropy (nats)")
	plot!(horizons, (mean(entropies_AIF,dims = 2)),yerror = std(entropies_AIF,dims = 2)./(sqrt(length(entropies_AIF[1,:]))), label = "EFE agent",ylim = (0,10),linewidth = 2.5)
	plot!(horizons,entropy_MOP.*ones(length(horizons)),label = "MOP agent")
	savefig("AIF/entropies.pdf")
end

# ╔═╡ 9c32afb3-1ea2-4206-846b-e674c1aac929
hs_u

# ╔═╡ c71b27fd-4ecd-4643-8a2b-c5c5d3a9335a
times_h,times_q,times_rand

# ╔═╡ 567f6b5d-c67e-4a43-9699-5625f1cc21a4
md"### Histogram of visited external states"

# ╔═╡ c9cfab90-ce27-40a6-b257-ec17fbd2d348
function plot_single_histogram(allx,ally,env;maxclim = 0.2,cbar = false)
	p_hist = plot(ticks = false)#, title = "H agent")#,ylabel = "\$u_{max} = 50\$")
	#Paint the whole arena 
	heatmap!(p_hist,zeros(env.sizex,env.sizey))
	#Draw obstacles
	plot!(p_hist,env.obstaclesx,env.obstaclesy,bins = (collect(0.5:1:env.sizex + 0.5),collect(0.5:1:env.sizey + 0.5)), st = :histogram2d, color = "#464646",clim = (0,maxclim))
	#Draw histogram
	histogram2d!(p_hist,allx,ally, bins = (collect(0.5:1:env.sizex + 0.5),collect(0.5:1:env.sizey + 0.5)),normalize = :probability,clim = (0,maxclim),cbar = cbar)
end

# ╔═╡ c933c84c-f158-4626-850b-7f5a164ea4aa
function plot_histogram(h_allx,h_ally,q_allx,q_ally,rand_allx,rand_ally,env1,t_hist = 500000,maxclim = 0.2)
	## H agent
	p_h_hist = plot(ticks = false)#, title = "H agent")#,ylabel = "\$u_{max} = 50\$")
	#Paint the whole arena 
	heatmap!(p_h_hist,zeros(env1.sizex,env1.sizey))
	#Draw obstacles
	plot!(p_h_hist,env1.obstaclesx,env1.obstaclesy,bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)), st = :histogram2d, color = "#464646",clim = (0,maxclim))
	#Draw histogram
		histogram2d!(p_h_hist,h_allx,h_ally, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :probability,clim = (0,maxclim),cbar = false)

	## R agent
	p_q_hist = plot(ticks = false)#, title = "R agent, ϵ = $(ϵ)")
	#Paint the whole arena
	heatmap!(p_q_hist,zeros(env1.sizex,env1.sizey))
	#Draw obstacles
	plot!(p_q_hist,env1.obstaclesx,env1.obstaclesy,bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)), st = :histogram2d, color = "#464646",clim = (0,maxclim))
	#Draw histogram
	histogram2d!(p_q_hist, q_allx,q_ally, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :probability,cbar = false,clim = (0,maxclim))

	## Random walker
	p_rand_hist = plot(ticks = false)#, title = "Random walker")
	#Paint the whole arena
	heatmap!(p_rand_hist,zeros(env1.sizex,env1.sizey))
	#Draw obstacles
	plot!(p_rand_hist,env1.obstaclesx,env1.obstaclesy,bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)), st = :histogram2d, color = "#464646",clim = (0,maxclim))
	#Draw histogram
	histogram2d!(p_rand_hist, rand_allx,rand_ally, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :probability,clim = (0,maxclim))
	
	#Colorbar
	plot!(p_rand_hist,colorbar_ticks = collect(maxclim/4:maxclim/4:maxclim))
	# p_h200_hist = plot(ticks = false)
	# #plot!(p_hist1,env1.obstaclesx,env1.obstaclesy, st = :histogram2d)
	# heatmap!(p_h200_hist,zeros(env1.sizex,env1.sizey))
	# histogram2d!(p_h200_hist,h_allx_200,h_ally_200, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = true,clim = (0,0.2),cbar = false,ylabel = "\$u_{max} = 200\$")
	# p_q200_hist = plot(ticks = false)
	# heatmap!(p_q200_hist,zeros(env1.sizex,env1.sizey))
	# histogram2d!(p_q200_hist, q_allx_200,q_ally_200, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = true,clim = (0,0.2))
	p_h_hist,p_q_hist,p_rand_hist
	end

# ╔═╡ 69a5d51e-84dd-4d81-8f26-19d4eb8cc9bc
begin
	p_h_hist_fa2,p_KL_hist_fa2,p_KL_hist_va2 = plot_histogram(h_xpos_fa,h_ypos_fa,q_xpos_fa,q_ypos_fa,[states_KL_va[i][1] for i in 1:length(states_KL_va)],[states_KL_va[i][2] for i in 1:length(states_KL_va)],env1,max_t,0.03)
	plot!(p_h_hist_fa2,title = "MOP, fixed actions")
	plot!(p_KL_hist_fa2,title = "R, fixed actions")
	plot!(p_KL_hist_va2,title = "KL, fixed or variable actions")
	plot(p_h_hist_fa2,p_KL_hist_fa2,p_KL_hist_va2, layout = Plots.grid(1, 3, widths=[0.3,0.3,0.4]),size = (1000,300),margin = 5Plots.mm)
	#savefig("locations_histogram_MOP_R_fixedactions.png") 
end

# ╔═╡ 0714bc63-22d9-4f7f-a2d7-b4292575e05c
begin
	p_h_hist_fa,p_KL_hist_va,p_KL_hist_fa = plot_histogram(h_xpos_fa,h_ypos_fa,[states_KL_va[i][1] for i in 1:length(states_KL_va)],[states_KL_va[i][2] for i in 1:length(states_KL_va)],[states_KL_fa[i][1] for i in 1:length(states_KL_fa)],[states_KL_fa[i][2] for i in 1:length(states_KL_fa)],env1,max_t,0.03)
	plot!(p_h_hist_fa,title = "MOP, fixed actions")
	plot!(p_KL_hist_fa,title = "KL, fixed actions everywhere")
	plot!(p_KL_hist_va,title = "KL, fixed actions")
	plot(p_h_hist_fa,p_KL_hist_fa2,p_KL_hist_va,p_KL_hist_fa, layout = Plots.grid(1, 4, widths=[0.22,0.22,0.22,0.34]),size = (1300,300),margin = 5Plots.mm)
	#savefig("locations_histogram_MOP_R_KL.pdf") 
end

# ╔═╡ e671cb3e-2d1a-4196-9274-89d41ac323c8
begin
	p_h_hist,p_q_hist,p_rand_hist=plot_histogram(hs_x,hs_y,qs_x,qs_y,rand_x,rand_y,env1,max_t,0.015)
	plot(p_h_hist,p_q_hist,p_rand_hist, layout = Plots.grid(1, 3, widths=[0.3,0.3,0.4]),size = (1000,300),margin = 5Plots.mm)
	#savefig("locations_histogram_threeagents_nepisodes_$(n_episodes_hist)_gain_$(food_gain).pdf") 
end

# ╔═╡ b703ecd5-79e5-4c4c-8e7a-4d96ffa9942d
begin
	p_h_hist2,p_q_hist2,p_AIF_hist = plot_histogram(hs_x,hs_y,[states_emp[i][1] for i in 1:length(states_emp)],[states_emp[i][2] for i in 1:length(states_emp)],[states_AIF[i][1] for i in 1:length(states_AIF)],[states_AIF[i][2] for i in 1:length(states_AIF)],env1,max_t,0.015)
	plot!(p_h_hist2,title = "MOP")
	plot!(p_q_hist2,title = "$(p_emp.n)-step MPOW")
	plot!(p_AIF_hist,title = "H = $(p_AIF.H), \$\\lambda\$ = $(p_AIF.β) EFE")
	plot(p_h_hist2,p_q_hist2,p_AIF_hist, layout = Plots.grid(1, 3, widths=[0.3,0.3,0.4]),size = (1000,300),margin = 5Plots.mm)
	#savefig("locations_histogram_MOP_MPOW_AIF_nepisodes_$(n_episodes_hist)_gain_$(food_gain)_horizon_$(p_AIF.H)_beta_$(p_AIF.β).pdf") 
end

# ╔═╡ a372e0e5-7c03-4db5-b13f-700685d972d0
md"## Comparison with empowerment"

# ╔═╡ 9fa11207-16ae-45aa-b42c-95d7813817fa
begin
	plot_emp = plot_single_histogram([states_emp[i][1] for i in 1:length(states_emp)],[states_emp[i][2] for i in 1:length(states_emp)],env_emp,cbar = true,maxclim = 0.4)
	plot(plot_emp,title = "$(p_emp.n)-step empowerment", size = (300,300),margins = 5Plots.mm)
end

# ╔═╡ b2918c10-ca06-4c1d-91c1-c17e4dd49c9d
md"### Animations? $(@bind animations CheckBox(default = false))"

# ╔═╡ 3e8a7dbb-8dfa-44f8-beab-ea32f3d478b4
function animate_histogram(h_allx_50,h_ally_50,q_allx_50,q_ally_50,env1,maxclim,tstep = 10000,max_t = 500000)
	anim = @animate for t_hist in 1:tstep:Int(max_t)
	## H agent
	p_h50_hist = plot(ticks = false, title = "H agent")#,ylabel = "\$u_{max} = 50\$")
	#Paint the whole arena 
	heatmap!(p_h50_hist,zeros(env1.sizex,env1.sizey))
	#Draw obstacles
	plot!(p_h50_hist,env1.obstaclesx,env1.obstaclesy,bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)), st = :histogram2d, color = "black")
	#Draw histogram
		histogram2d!(p_h50_hist,h_allx_50[1:t_hist],h_ally_50[1:t_hist], bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :false,cbar = false,clim = (0,maxclim))

	## Q agent
	p_q50_hist = plot(ticks = false, title = "Q agent")
	#Paint the whole arena
	heatmap!(p_q50_hist,zeros(env1.sizex,env1.sizey))
	#Draw obstacles
	plot!(p_q50_hist,env1.obstaclesx,env1.obstaclesy,bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)), st = :histogram2d, color = "black")
	histogram2d!(p_q50_hist, q_allx_50[1:t_hist],q_ally_50[1:t_hist], bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :false,clim = (0,maxclim))
	
	# p_h200_hist = plot(ticks = false)
	# #plot!(p_hist1,env1.obstaclesx,env1.obstaclesy, st = :histogram2d)
	# heatmap!(p_h200_hist,zeros(env1.sizex,env1.sizey))
	# histogram2d!(p_h200_hist,h_allx_200,h_ally_200, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = true,clim = (0,0.2),cbar = false,ylabel = "\$u_{max} = 200\$")
	# p_q200_hist = plot(ticks = false)
	# heatmap!(p_q200_hist,zeros(env1.sizex,env1.sizey))
	# histogram2d!(p_q200_hist, q_allx_200,q_ally_200, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = true,clim = (0,0.2))
	plot(p_h50_hist,p_q50_hist, layout = Plots.grid(1, 2, widths=[0.45,0.55]),size = (800,400),margin = 5Plots.mm)
	end
	#plot(p_h50_hist,p_q50_hist,p_h200_hist,p_q200_hist, layout = Plots.grid(2, 2, widths=[0.45,0.55]),size = (800,700),margin = 4Plots.mm)
	#savefig("histogram_space_fourrooms_$(env1.sizex).svg")
	anim
end

# ╔═╡ 298d4faf-bca5-49a5-b0e8-9e0b3600e3ae
begin
	if animations == true
		maxclim = 30000
		tstep = 5000
		maxt = max_t
		hist_animated = animate_histogram(h_allx_50,h_ally_50,q_allx_50,q_ally_50,env1,maxclim,tstep,maxt)
	end
end

# ╔═╡ cc3af52b-1d47-48b5-bbc4-9ef1327f4dfa
#gif(hist_animated,fps = 5,"histograms_animated.gif")

# ╔═╡ 1ace6a77-cc48-4b3d-8774-035015ffd74a
function animate_trajectory(h_x_pos,h_y_pos,q_x_pos,q_y_pos,env,tstep = 1,max_t = 500000)
	
	## H agent
	p_h50 = plot(ticks = false, title = "H agent")#,ylabel = "\$u_{max} = 50\$")
	p_q50 = plot(ticks = false, title = "Q agent")
	reward_sizes = env.reward_mags
	#Draw obstacles
	scatter!(p_h50,env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 15, color = "black")
	scatter!(p_q50,env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 15, color = "black")
	#Draw food
	for i in 1:length(reward_mags)
		scatter!(p_h50,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i]*2,color = "green",markershape = :diamond)
		scatter!(p_q50,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i]*2,color = "green",markershape = :diamond)
	end
	plot!(p_h50, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	plot!(p_q50, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))

	anim = @animate for t_traj in tstep:tstep:Int(max_t)

	#Draw histogram
		plot!(p_h50, h_x_pos[t_traj-tstep+1:t_traj+1],h_y_pos[t_traj-tstep+1:t_traj+1], markersize = 1, leg = false, color = "blue", linealpha = 0.2)

	## Q agent
	plot!(p_q50, q_x_pos[t_traj-tstep+1:t_traj],q_y_pos[t_traj-tstep+1:t_traj], markersize = 1, leg = false, color = "blue", linealpha = 0.2)
	plot(p_h50,p_q50, layout = Plots.grid(1, 2, widths=[0.5,0.5]),size = (800,400),margin = 3Plots.mm)
	end
	#plot(p_h50_hist,p_q50_hist,p_h200_hist,p_q200_hist, layout = Plots.grid(2, 2, widths=[0.45,0.55]),size = (800,700),margin = 4Plots.mm)
	#savefig("histogram_space_fourrooms_$(env1.sizex).svg")
	anim
end

# ╔═╡ 19b1a5c0-c5a8-45bc-8cf0-fc6f97ccff91
begin
	if animations == true
		tstep_traj = 5000
		maxt_traj = max_t
		traj_animated = animate_trajectory(h_allx_50,h_ally_50,q_allx_50,q_ally_50,env1,tstep_traj,maxt_traj)
	end
end

# ╔═╡ cda6ab20-3dbb-43eb-9ae3-71889cb3da88
#gif(traj_animated,fps = 5,"trajectories.gif")

# ╔═╡ 1483bfe5-150d-40f5-b9dc-9488dcbc88b2
md"### Histogram of visitation of internal states"

# ╔═╡ e9e0c092-53c8-4fb0-b116-4849c2ff63c0
begin
	_,_,h_us,_,_,_,_,_,_ = sample_trajectory(s_0,u_0,h_value,max_t,env1,params,false)
	_,_,q_us,_,_,_,_,_ = sample_trajectory_q(s_0,u_0,q_value,ϵ,max_t,env1,params,false)
end

# ╔═╡ d93ef435-c460-40cc-96e7-817e9eaace55
begin
	p_us = plot(xlim = (0,100), xlabel = "Internal energy", ylabel = "Probability",legend_foreground_color = nothing, margin = 4Plots.mm, majorgrid = false,size = (400,300),minorgrid = false)
	bd = 1
	plot!(p_us,h_us, label = "MOP agent", bandwidth = bd, st = :density,linewidth = 3)
	plot!(p_us,q_us, label = "R agent", bandwidth = bd, st = :density,linewidth = 3)
	plot!(p_us,us_emp, label = "E agent", bandwidth = bd, st = :density,linewidth = 3)
	# plot!(p_us,h_us_50, label = "H agent", bins = collect(0:1:50), st = :stephist,normalized = :pdf, linewidth = 3)
	# plot!(p_us,q_us_50, label = "Q agent", bins = collect(0:1:50), st = :stephist,normalized = :pdf, linewidth = 3)
	#savefig("energies_histogram.pdf")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
ZipFile = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"

[compat]
Distributions = "~0.25.100"
Parameters = "~0.12.3"
Plots = "~1.39.0"
PlutoUI = "~0.7.52"
StatsBase = "~0.34.0"
StatsPlots = "~0.15.6"
ZipFile = "~0.10.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "7ed3ab3e99a0889593c6cf6adf05ca2cc9e958f7"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

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
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
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
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.ChangesOfVariables]]
deps = ["InverseFunctions", "LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "b86ac2c5543660d238957dbde5ac04520ae977a7"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

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
git-tree-sha1 = "b6def76ffad15143924a2199f72a5cd883a2e8a9"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.9"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "938fe2981db009f531b6332e31c58e9584a2f9bd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.100"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

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
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "a20eaa3ad64254c61eeb5f230d9306e937405434"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.6.1"

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
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

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
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

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
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

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
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "68772f49f54b479fa88ace904f6127f0a3bb2e46"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.12"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

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
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

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

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

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
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

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
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

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
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "0d097476b6c381ab7906460ef1ef1638fbce1d91"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.2"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

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
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

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
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "bf6085e8bd7735e68c210c6e5d81f9a6fe192060"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.19"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "7c29f0e8c575428bd84dc3c72ece5178caa67336"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.2+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

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
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

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

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "d5fb407ec3179063214bc6277712928ba78459e2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a1f34829d5ac0ef499f6d84428bd6b4c71f02ead"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

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
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

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

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "04a51d15436a572301b5abbb9d099713327e9fc4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.4+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

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

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

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

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═a649da15-0815-438d-9bef-02c6d204656e
# ╠═25c56490-3b9c-4825-b91d-8b9e41fc0f6b
# ╠═99246057-6b37-4d85-ae77-2a4210fac365
# ╠═422493c5-8a90-4e70-bd06-40f8e6b254f1
# ╠═76f77726-7776-4975-9f30-3887f13ae3e7
# ╠═393eaf2d-e8fe-4675-a7e6-32d0fe9ac4e7
# ╠═b4e7b585-261c-4044-87cc-cbf669768145
# ╟─7feeec1a-7d7b-4220-917d-049f1e9b101b
# ╠═7e68e560-45d8-4429-8bff-3a8229c8c84e
# ╠═a41419bd-1859-4a08-8ce0-a5476e256284
# ╠═986f5441-9361-4074-a7f6-7affe650e555
# ╟─194e91cb-b619-4908-aebd-3136107175b7
# ╟─a46ced5b-2e58-40b2-8eb6-b4840043c055
# ╠═9404080e-a52c-42f7-9abd-ea488bf7abc2
# ╠═0dcdad0b-7acc-4fc4-93aa-f6eacc077cd3
# ╠═0ce119b1-e269-41e2-80b7-991cae37cf5f
# ╠═8675158f-97fb-4222-a32b-49ce4f6f1d41
# ╠═ec119eb7-894c-4eb1-95ea-4b4dae9526c3
# ╟─92bca36b-1dc9-4c03-88c0-6a684dfbec9f
# ╠═c96e3331-1dcd-4b9c-b28d-d74493c8934d
# ╟─d0a5c0fe-895f-42d8-9db6-3b0fcc6bb43e
# ╠═155056b5-21ea-40d7-8cce-19fde5a1b150
# ╟─6c716ad4-23c4-46f8-ba77-340029fcce87
# ╠═07abd5b7-b465-425b-9823-19b73d07db56
# ╠═8f2fdc23-1b82-4479-afe7-8eaf3304a122
# ╟─403a06a7-e30f-4aa4-ade1-55dee37cd514
# ╠═bd16a66c-9c2f-449c-a792-1073c54e990b
# ╟─ac3a4aa3-1edf-467e-9a47-9f6d6655cd04
# ╟─c6870051-0241-4cef-9e5b-bc876a3894fa
# ╠═a6fbddc0-7def-489e-9ad1-2e6a5e68eddd
# ╠═d88e0e27-2354-43ad-9c26-cdc90beeea0f
# ╟─184636e2-c87d-4a89-b231-ff4aef8424d5
# ╠═82fbe5a0-34a5-44c7-bdcb-36d16f09ea7b
# ╠═a11b198f-0a55-4529-b44c-270f37ef773a
# ╠═73722c01-adee-4bfd-97b4-60f2ced23725
# ╠═25e35560-5d80-4388-8002-fd29d0541b18
# ╠═e67db730-ca7c-4ef4-a2d2-7e001d5f7a79
# ╠═76f506dc-b21d-4e13-a8e8-9d1b3bd21b30
# ╟─aa5e5bf6-6504-4c01-bb36-df0d7306f9de
# ╟─ef9e78e2-d61f-4940-9e62-40c6d060353b
# ╟─a4457d71-27dc-4c93-81ff-f21b2dfed41d
# ╟─7ad00e90-3431-4e61-9a7f-efbc14d0724e
# ╠═a12ae2fc-07d4-459e-8459-747ab12fdbd5
# ╟─b072360a-6646-4d6d-90ea-716085c53f66
# ╠═a0729563-0b6d-4014-b8c7-9eb284a34606
# ╠═11b5409c-9db8-4b34-a111-7a62fedd23be
# ╟─f98d6ea0-9d98-4940-907c-455397158f3b
# ╟─5f4234f5-fc0e-4cdd-93ea-99b6463b2ba1
# ╠═7a0173ac-240d-4f93-b413-45c6af0f4011
# ╠═27011f44-929a-4899-b822-539d270959e1
# ╠═caadeb3b-0938-4559-8122-348c960a6eb1
# ╠═29a4d235-8b03-4701-af89-cd289f212e7d
# ╠═819c1be2-339f-4c37-b8a3-9d8cb6be6496
# ╟─358bc5ca-c1f6-40f1-ba2d-7e8466531903
# ╠═40d62df0-53bb-4b46-91b7-78ffd621a519
# ╟─005720d3-5920-476b-9f96-39971f512452
# ╠═2379dcc3-53cb-4fb6-b1e8-c851e36acd1f
# ╟─6e7b6b2a-5489-4860-930e-47b7df014840
# ╟─2ed5904d-03a3-4999-a949-415d0cf47328
# ╠═787bbe73-6052-41e0-bc8c-955e4a884886
# ╠═139c806d-3f52-4fb9-9fe8-c57259ed1b6f
# ╠═6a29cc32-6abf-41c1-b6e3-f4cb33b76f46
# ╠═a5f43388-9e45-497f-b0f8-7f2987b3102d
# ╠═293c129d-dba4-4b04-aa0a-66e1b570aad4
# ╟─ba9a241e-93b7-42fd-b790-d6010e2435bd
# ╠═39b00e87-db6e-441f-8265-de75148d6519
# ╠═a6e21529-00eb-40b8-9f40-b8b26171eaad
# ╠═c8c70b3f-45d5-4d38-95af-85ae85d1233c
# ╟─a14df073-5b27-4bdb-b4c8-03927909b12e
# ╠═ad2855ee-ffc3-4910-a307-2e908b93706b
# ╠═5b3bbb53-d5f7-462b-8425-c8c6c4ff3865
# ╠═91826a58-ebd5-4e2f-8836-ad6cff22406e
# ╠═828b1bcd-a093-463c-bb2f-4c1b5359939c
# ╠═c7bc355d-d8dd-4b82-ac81-1b9a6d5f71e0
# ╠═af848419-950a-4827-bc4e-fc3e8facd1a8
# ╠═f405c26e-f911-4dea-9364-f751259acf43
# ╠═cb3280eb-8453-4040-aac0-22c1dc93b9d9
# ╠═01606177-e88b-453c-ae34-8b3f5755bb82
# ╠═4f5d1b29-90a2-4b79-9ce0-30764e6d3350
# ╠═4b5eaa5a-df95-4060-bb99-1e69ccaf8533
# ╠═e1c59a22-fdd8-4edd-86d5-9efd9aca69a5
# ╠═e3ff07ca-6c46-4be9-bf78-3c09bed248f6
# ╠═55fdc810-fb0c-4d63-8010-f65200bceb5d
# ╠═0714bc63-22d9-4f7f-a2d7-b4292575e05c
# ╠═69a5d51e-84dd-4d81-8f26-19d4eb8cc9bc
# ╟─7431c898-bbef-422e-af13-480d2e912d16
# ╟─86685690-c31d-4b71-b070-43fdb18e48db
# ╠═f2f45f6f-c973-4272-afa2-7077dd169790
# ╠═0de63f0b-c545-445b-a0da-97ec75647598
# ╠═c555b1ac-ed4f-4767-9680-0e7ef7ab758f
# ╟─33df863c-bf8c-4624-9fd3-1d3a569a4f61
# ╟─01aaed4b-e028-49c9-90b4-a6c4158929f1
# ╠═49df9fba-22a8-48e9-8278-22ffdfc7765f
# ╠═3dd47db4-2485-4833-8172-3fe7a3fe2214
# ╟─e4dcc71c-4cc6-4b8f-88ad-7c2c18e5af02
# ╠═4121c28e-010e-428a-8e9a-544231dc6c0b
# ╠═1c5176b2-f7e0-4a9a-b07c-392840294aa0
# ╟─e2da5d6b-4134-4bee-a93b-a153d19359cf
# ╠═e46e0035-c649-485f-9e87-6f7231c0927a
# ╠═b1737188-4e57-4d81-a1fa-b55e1621a7dc
# ╠═73b6c6e3-0c3c-46d5-a109-1f512b6871ee
# ╠═8ec4f121-410a-49c6-8ae5-f8e014757fc7
# ╠═0a693ea2-9506-4217-86d7-514678c03104
# ╠═118e6de3-b525-42e5-8c0c-654c525a4684
# ╟─3014c91c-b705-42e1-9689-ebb1c357f8b2
# ╠═49124160-32b0-4e09-904c-9547697cbfd1
# ╠═521717ad-23b7-4bca-bc8d-a2387946e55d
# ╠═d6d26401-22eb-4e8d-8b2a-13223167a949
# ╠═d99990c1-1b0c-4d52-87a3-6b3bd363fde7
# ╟─00025055-7ec0-486d-bb29-8e769e08fcc8
# ╟─b7c66258-bb37-4660-9b22-783a50d4c6f8
# ╟─4f6cdf92-0821-4d39-ae15-1844a4a29482
# ╠═6edeefb3-186c-447a-b6e6-abab6a7d1c63
# ╠═317fab4f-c0b0-4671-a653-1bc8cbbfabc4
# ╠═29193a6f-737f-4efb-acaa-d16d2b8c3589
# ╠═0137a63c-6beb-4aaa-a485-f4b5e08fab01
# ╠═210ef48a-13b1-44b5-a4c6-c2084bafc778
# ╠═0a55db1e-4dc7-4195-aa2f-564d770afa8c
# ╠═38d916de-9068-4f37-9ace-81e115d731bc
# ╠═f0cb716c-620b-4632-9022-f7b9880ac98e
# ╠═6046a1ae-ec46-490c-baa7-1534f72c5ea9
# ╠═9ec3d452-5cc0-447d-8965-9b487f39650a
# ╠═51432185-5447-45a5-8734-f85ad54b5602
# ╠═48ade83b-bd73-4570-9a14-590f6f846097
# ╠═d51573c2-f8b8-4785-bc14-3f611a34a924
# ╠═c515caee-858d-4e1f-b021-7dd5b8648549
# ╠═df4ce514-884c-401a-85f3-02ec37444816
# ╠═446c2897-8db9-452d-9d11-4cf7cc9bfa8a
# ╠═89c24ce1-87ce-4145-b3e7-bc6da50b94bf
# ╠═8dcb6536-3737-4d58-862e-094218874040
# ╠═4d96f296-b22a-4243-b32e-7154d88a16d7
# ╠═a8152784-c97f-4937-8791-166a411eee80
# ╠═e9b5ab32-0899-4e3d-b1cb-c986ae82b12d
# ╠═b8ce5e63-5f88-424f-8d78-0910b0f88762
# ╠═a12232f3-0c6b-44c2-bcba-f150a1f39c88
# ╠═188dc342-f0ae-4df1-adf6-bce3da9d5d76
# ╠═b92cde2a-bc29-4417-8a88-0f527a1b790e
# ╟─d801413b-adff-48f0-aa90-89a1af1c0d63
# ╟─2581ecba-d9c1-4989-b718-4f559c870adb
# ╠═93980ec8-f9d2-4637-945c-259715e3ef5d
# ╟─0fdb6f27-479b-4b46-bea2-2b6158f3d1c7
# ╟─164985f1-3dda-4ab1-b3d2-af827eace611
# ╟─df146a19-b47b-49eb-993a-7233df3741aa
# ╠═d30c86b5-61eb-4e79-a485-f0246cae4064
# ╠═f5313ab6-e354-4a9e-837f-a03b09b08d1e
# ╟─0beeaba3-e0a7-4975-8e50-0c72ca3df314
# ╠═4c79a36c-e679-4fe1-a036-e19f649f4997
# ╟─0f656395-ffc7-4010-abb9-04b43121bcfb
# ╠═52a823a0-4fb9-4fcf-87d6-beecc2cb1ab2
# ╠═b38a0ef8-4889-4f56-a41e-8d5173c5db50
# ╠═f66c8ed5-db75-45f0-9962-e63a04caae80
# ╠═58b6f009-bd69-4bb7-bd44-c395718fae5d
# ╠═84b2cdf6-1f82-4493-9aa7-97ea74bd9592
# ╟─9eff1101-d2f8-4952-8efe-d8e6ce9bc195
# ╟─496c1dbe-1052-45c1-9448-31befea96222
# ╟─586a8b9b-1a02-41a0-93c0-85e69f56da90
# ╟─6ad087b7-c848-498b-beba-fbd5c4b0c4c4
# ╠═867527c2-e09e-4486-ba06-99a83fda13c6
# ╠═5bebc6e9-1bde-40f4-bac4-2f5ec05ab548
# ╟─d2df50cb-d163-4c49-b940-35179b7148a5
# ╠═ffd1a342-c46e-4016-a24b-fc98e4498890
# ╠═3fb25daf-c9af-4746-b8a6-91bc01e4d12b
# ╠═9f11779e-c23e-41d3-a51d-eb88db85c7fd
# ╠═a134c7d5-15e6-4a22-bc34-56a3155b88dd
# ╠═f2275b68-335a-4383-9ad9-b2e47286f008
# ╟─53e6d8b6-fe76-4b1a-b6e4-8c55fc58db7d
# ╠═d66ab0d0-249d-464b-92d7-a0493338b7d0
# ╠═759ae341-ec69-4557-a845-cfcbcedb26dd
# ╠═b63f2611-0bca-4e68-bd1d-dc9cef75d9f1
# ╠═20cba048-e6c8-42d3-ad15-8091a8fd9bfc
# ╠═9b45f045-77d6-43c1-a7a0-5d4d5c7af480
# ╟─e7578dbf-ac6c-414c-9e08-1ed9636177f7
# ╠═6b3fb79f-be03-4d90-9527-83e868cdaddd
# ╟─5b1ba4f6-37a9-4e61-b6cc-3d495aa67c9d
# ╟─a0dc952c-e733-41de-8d6e-458d66c3769a
# ╠═b17456b8-3d6a-4c31-95f2-036c5fd90ea3
# ╠═752633e4-ed90-4cab-819f-af03bcf890c7
# ╟─37d7baa7-b22b-4919-8910-43292287b230
# ╠═cb8b2b44-13ee-4c12-833a-26f6d806724b
# ╟─a14c2f1f-ad38-45e5-93a9-d6f19641687b
# ╠═36276897-e845-46b4-8883-a8255d5ecc9c
# ╠═87a66b34-e7e3-4d3b-9da1-089ad296c643
# ╠═eef8a58c-4163-49a2-bbdf-48f9f2782a08
# ╠═dd9005fe-8777-4f71-aa7e-8de959dcbc52
# ╠═ffaa2c35-f6c9-45de-b67c-861dd1aec914
# ╠═fd4463de-56de-4707-8a17-f5be5a0cd33e
# ╠═dc8dd49b-cb51-45cc-8cd7-ffa497b3f153
# ╠═676487d4-839b-4641-ad77-af3f1610787f
# ╟─91e0e0ca-d9df-44d7-81e6-00b343ad9bf0
# ╠═1bb9994a-ed89-4e08-921e-39d46fc45e0a
# ╠═beb1a211-f262-49d8-a3c4-73a0cf727921
# ╠═5d0ad59b-366e-4660-9350-92d34d616f16
# ╟─c7d270aa-9c5c-461b-ac6b-2b9287a2d461
# ╟─5cc21644-35f9-404c-8f68-c1591a19226e
# ╠═aea98aa9-46a5-47d7-b78f-3b784bcb8668
# ╠═8e1241e7-6a23-46cb-8b8b-a6c828fd6f17
# ╟─1767238d-9301-4e84-8b9c-54b8369065d9
# ╠═f79a8a77-02f5-4677-a34f-e6a23be4d5ad
# ╠═ebe880f4-7001-4118-b523-037af729d95f
# ╠═8a6d0254-211f-45dd-8798-8408cb9ccc55
# ╠═96a54510-789e-455c-bbab-390f78e909cd
# ╠═d2f49d6b-2acc-4fef-8e24-9acdd7591977
# ╠═6dcaa4f7-a539-412f-8ee0-f1f9dfba831f
# ╠═94430097-f64f-4d27-aa0f-24aec2042d5d
# ╠═abb61da7-3711-4fa6-a021-2bb916411019
# ╟─1c69c08b-aae5-453e-98ef-df35c7b4db50
# ╠═0a9dc717-dbcb-4b27-8c76-cb8fbfdbec96
# ╟─1f3be0d8-d296-4eb0-a951-bd862914ae92
# ╠═bb45134a-88b1-40d2-a486-c7afe8ac744e
# ╠═f1d5ee65-10f5-424a-b018-2aff1a5d7ff8
# ╠═78a5caf6-eced-4783-b950-26563f632be2
# ╠═4a868ec2-b636-4d5d-a248-0a4e0cca3668
# ╠═6edb5b48-b590-492d-bd3e-6a4f549aae30
# ╠═4c51afa2-e294-4922-9cae-d087582d771c
# ╠═c842a822-462b-4849-86f1-3fc717c492c1
# ╠═42423e46-2446-4cdd-84c1-fe824e5eb3c4
# ╠═d3a2593d-0d55-470c-bff1-4d80714a6f3a
# ╠═9c32afb3-1ea2-4206-846b-e674c1aac929
# ╠═c71b27fd-4ecd-4643-8a2b-c5c5d3a9335a
# ╟─567f6b5d-c67e-4a43-9699-5625f1cc21a4
# ╟─c9cfab90-ce27-40a6-b257-ec17fbd2d348
# ╟─c933c84c-f158-4626-850b-7f5a164ea4aa
# ╠═e671cb3e-2d1a-4196-9274-89d41ac323c8
# ╠═b703ecd5-79e5-4c4c-8e7a-4d96ffa9942d
# ╟─a372e0e5-7c03-4db5-b13f-700685d972d0
# ╠═9fa11207-16ae-45aa-b42c-95d7813817fa
# ╟─b2918c10-ca06-4c1d-91c1-c17e4dd49c9d
# ╟─3e8a7dbb-8dfa-44f8-beab-ea32f3d478b4
# ╠═298d4faf-bca5-49a5-b0e8-9e0b3600e3ae
# ╠═cc3af52b-1d47-48b5-bbc4-9ef1327f4dfa
# ╟─1ace6a77-cc48-4b3d-8774-035015ffd74a
# ╠═19b1a5c0-c5a8-45bc-8cf0-fc6f97ccff91
# ╠═cda6ab20-3dbb-43eb-9ae3-71889cb3da88
# ╟─1483bfe5-150d-40f5-b9dc-9488dcbc88b2
# ╠═e9e0c092-53c8-4fb0-b116-4849c2ff63c0
# ╠═d93ef435-c460-40cc-96e7-817e9eaace55
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
