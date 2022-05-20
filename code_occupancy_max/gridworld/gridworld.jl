### A Pluto.jl notebook ###
# v0.19.4

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

# ╔═╡ 422493c5-8a90-4e70-bd06-40f8e6b254f1
gr()

# ╔═╡ 76f77726-7776-4975-9f30-3887f13ae3e7
default(titlefont = ("Computer Modern",16), legend_font_family = "Computer Modern", legend_font_pointsize = 14, guidefont = ("Computer Modern", 16), tickfont = ("Computer Modern", 14))

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
	γ::Float64 = 0.99
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

# ╔═╡ 194e91cb-b619-4908-aebd-3136107175b7
function adm_actions(s_state,u_state,env::environment, constant_actions = false)
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
	s_prime
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
md" # Environment"

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

# ╔═╡ 8f2fdc23-1b82-4479-afe7-8eaf3304a122
begin
	size_x = 11
	size_y = 11
	capacity = 100
	food_gain = 10
	reward_locations = [[1,1],[size_x,size_y],[size_x,1],[1,size_y]]
	reward_mags = [1,1,1,1].*food_gain
	env1 = initialize_fourrooms(size_x,size_y,capacity,reward_locations,reward_mags)
end

# ╔═╡ 403a06a7-e30f-4aa4-ade1-55dee37cd514
function draw_environment(x_pos,y_pos,u,env::environment)
	reward_sizes = env.reward_mags
	#Draw obstacles
	ptest = scatter(env.obstaclesx,env.obstaclesy,markershape = :square, markersize = 20, color = "black")
	#Draw food
	for i in 1:length(reward_mags)
		scatter!(ptest,[env.reward_locations[i][1]],[env.reward_locations[i][2]],markersize = reward_sizes[i]*2,color = "green",markershape = :diamond)
	end
	#Draw agent
	plot!(ptest, gridalpha = 0.8,  xticks = collect(0.5:env.sizex+0.5), yticks = collect(0.5:env.sizey+0.5), tickfontcolor = :white, grid = true, ylim=(0.5,env.sizey +0.5), xlim=(0.5,env.sizex +0.5))
	#Draw arrows
	actions,_ = adm_actions([x_pos[1],y_pos[1]],u[1],env)
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
function h_iteration(env::environment,tolerance = 1E-2, n_iter = 100,verbose = false)
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
					for a in actions
						s_primes = reachable_states(s,a)
						for s_prime in s_primes
							u_prime = transition_u(s,u,a,env)
							Z += exp(env.γ*value_old[s_prime[1],s_prime[2],u_prime])
						end
					end
					value[x,y,u] = log(Z)
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

# ╔═╡ d88e0e27-2354-43ad-9c26-cdc90beeea0f
function optimal_policy(s,u,optimal_value,env::environment,verbose = false)
	actions,_ = adm_actions(s,u,env)
	policy = zeros(length(actions))
	Z = exp(optimal_value[s[1],s[2],u])
	#Only compute policy for available actions
	for (idx,a) in enumerate(actions)
		u_p = transition_u(s,u,a,env)
		s_p = transition_s(s,a,env)
		policy[idx] = exp(env.γ*optimal_value[s_p[1],s_p[2],u_p]-optimal_value[s[1],s[2],u])
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
		

# ╔═╡ 184636e2-c87d-4a89-b231-ff4aef8424d5
md"## Optimal value function"

# ╔═╡ 82fbe5a0-34a5-44c7-bdcb-36d16f09ea7b
begin
	#To compute
	#h_value,t_stop = h_iteration(env1,tol,n_iter,true)
	#Read out values
	#h_value = reshape(readdlm("values/h_value_gain_$(env1.reward_mags[1]).dat"),env1.sizex,env1.sizey,env1.sizeu)
	#Specific one
	h_value = reshape(readdlm("values/h_value_gain_$(food_gain).dat"),env1.sizex,env1.sizey,env1.sizeu)
end;

# ╔═╡ a11b198f-0a55-4529-b44c-270f37ef773a
#Write value function to file
#writedlm("h_value_u_$(env1.sizeu).dat",h_value)

# ╔═╡ e67db730-ca7c-4ef4-a2d2-7e001d5f7a79
u = 20
#@bind u PlutoUI.Slider(1:env1.sizeu)

# ╔═╡ 73722c01-adee-4bfd-97b4-60f2ced23725
function plot_optimal_policy(p,u,opt_value,env::environment,constant_actions = false)
	for x in 1:env.sizex
		for y in 1:env.sizey 
			ids = findall(i -> i == [x,y],env.obstacles)
			if length(ids) == 0
				actions,probs = optimal_policy([x,y],u,opt_value,env,false)
				arrow_x = zeros(length(actions))
				arrow_y = zeros(length(actions))
				aux = actions.*probs
				for i in 1:length(aux)
					arrow_x[i] = aux[i][1]*1.5
					arrow_y[i] = aux[i][2]*1.5
				end
				quiver!(p,ones(Int64,length(aux))*x,ones(Int64,length(aux))*y,quiver = (arrow_x,arrow_y),color = "green",linewidth = 2)
				scatter!(p,ones(Int64,length(aux))*x + arrow_x, ones(Int64,length(aux))*y + arrow_y,markersize = probs*30, color = "red")
			end
		end
	end
end		

# ╔═╡ 76f506dc-b21d-4e13-a8e8-9d1b3bd21b30
begin
	p1 = heatmap(transpose(h_value[:,:,u]), title = "optimal value function, u = $u",clims = (minimum(h_value[1,:,u]),maximum(h_value[1,:,u])))
	for i in 1:length(env1.reward_mags)
		if reward_mags[i]  > 0
			col = "green"
		else
			col = "gray"
		end
		scatter!(p1,[env1.reward_locations[i][1]],[env1.reward_locations[i][2]], color = col, markersize = min(abs(env1.reward_mags[i]),50))
	end
	plot!(p1,size = (1000,800)) 
	plot_optimal_policy(p1,u,h_value,env1)
	#p2 = heatmap(val_stoch[:,:,u], title = "value function random walk")
	plot(p1,legend = false)
	#savefig("optimal_value_function.png")
end

# ╔═╡ aa5e5bf6-6504-4c01-bb36-df0d7306f9de
md"## Sample trajectory"

# ╔═╡ ef9e78e2-d61f-4940-9e62-40c6d060353b
function sample_trajectory(s_0,u_0,opt_value,max_t,env::environment,occupancies = false)
	xpositions = Any[]
	ypositions = Any[]
	u_states = Any[]
	all_x = Any[]
	all_y = Any[]
	urgency = Any[]
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
		actions,policy = optimal_policy(s,u,opt_value,env)
		idx = rand(Categorical(policy))
		action = actions[idx]
		for i in 1:length(actions)
			if policy[i] > 0.6
				push!(urgency,u)
			end
		end
		s_p = transition_s(s,action,env)
		u_p = transition_u(s,u,action,env)
		s = s_p
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
		if occupancies == true
			if length(unvisited_s_states) == 0
				return xpositions,ypositions,u_states,all_x,all_y,urgency,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,t
			end
		end
	end
	xpositions,ypositions,u_states,all_x,all_y,urgency,1-length(unvisited_s_states)/n_arena_states,1-length(unvisited_u_states)/env.sizeu,max_t
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
	for i in 1:length(reward_mags)
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

# ╔═╡ b072360a-6646-4d6d-90ea-716085c53f66
md"Produce animation? $(@bind movie CheckBox(default = false))"

# ╔═╡ f98d6ea0-9d98-4940-907c-455397158f3b
md"# R agent"

# ╔═╡ 5f4234f5-fc0e-4cdd-93ea-99b6463b2ba1
function reachable_rewards(s,u,a,env::environment;delta_reward = 1E-5)
	r = 1
	if in(env1.reward_locations).([s]) == [true]
		r += delta_reward
	end	
	if u < 2
		r = 0
	end
	r
end

# ╔═╡ 7a0173ac-240d-4f93-b413-45c6af0f4011
function q_iteration(env::environment,ϵ,tolerance = 1E-2, n_iter = 100,verbose = false)
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
					values = zeros(length(actions))
					for (id_a,a) in enumerate(actions)
						s_primes = reachable_states(s,a)
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
#q_value,t_stop_q = q_iteration(env1,ϵ,0.1,n_iter,true);
#To read out from file
#q_value = reshape(readdlm("values/q_value_gain_$(env1.reward_mags[1]).dat"),env1.sizex,env1.sizey,env1.sizeu);
#Specific one
q_value = reshape(readdlm("values/q_value_gain_$(food_gain)_eps_$(ϵ).dat"),env1.sizex,env1.sizey,env1.sizeu);

# ╔═╡ 29a4d235-8b03-4701-af89-cd289f212e7d
#Write R agent value function to file
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
function optimal_policy_q(s,u,value,ϵ,env::environment)
	actions,ids_actions = adm_actions(s,u,env)
	q_values = zeros(length(actions))
	policy = zeros(length(actions))
	for (idx,a) in enumerate(actions)
		s_primes = reachable_states(s,a)
		r = reachable_rewards(s,u,a,env)
		for s_p in s_primes
			#deterministic environment
			u_p = transition_u(s,u,a,env)
			q_values[idx] += r + env.γ*value[s_p[1],s_p[2],u_p]
		end
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
optimal_policy_q([3,3],20,q_value,ϵ,env1)

# ╔═╡ 2379dcc3-53cb-4fb6-b1e8-c851e36acd1f
function sample_trajectory_q(s_0,u_0,opt_value,ϵ,max_t,env::environment,occupancies = true)
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
		actions_at_s,policy,n_actions = optimal_policy_q(s,u,opt_value,ϵ,env)
		idx = rand(Categorical(policy))
		action = actions_at_s[idx]
		#action = rand(actions)
		s_p = transition_s(s,action,env)
		u_p = transition_u(s,u,action,env)
		s = s_p
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

# ╔═╡ 47cf89c9-bb93-498d-9881-f079399b98ba
function sample_trajectory_random(s_0,u_0,max_t,env::environment,occupancies = true)
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
		actions_at_s,_ = adm_actions(s,u,env)
		action = rand(actions_at_s)
		#action = rand(actions)
		s_p = transition_s(s,action,env)
		u_p = transition_u(s,u,action,env)
		s = s_p
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
#function to animate both agents at the same time, side by side
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

# ╔═╡ 5b1ba4f6-37a9-4e61-b6cc-3d495aa67c9d
md"# Comparison between agents"

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
	#This computes occupancies running n_episodes, each with a max duration of t_episode, for both Q and H agent. It takes up to 8 hours for t_episode = 1E6 and 50 episodes and 4 parallel threads
	t_ep = 50000
	n_ep = 50
	e_gains = [2,3,5,10,20,30,50,75,100]
	#epsilons for final paper
	ϵs_new = [0.0,0.45,0.5,1.0]
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
#cols = palette(:Dark2_8)

# ╔═╡ 6dcaa4f7-a539-412f-8ee0-f1f9dfba831f
indices_to_plot = [2,4]

# ╔═╡ 94430097-f64f-4d27-aa0f-24aec2042d5d
begin
	plot(ylabel = "Survived time steps", xlabel = "Food gain")#, xlim = (2,14),xticks = collect(2:10))
	plot!(e_gains,mean(surv_times_h,dims = 2),label = "H agent",lw = 4)
	# for i in indices_to_plot
	# 	plot!(e_gains,mean(surv_times_q[:,i,:],dims=2), yerror = std(surv_times_q[:,i,:],dims = 2)./sqrt(n_ep),c = cols[i], label = "ϵ = $(ϵs_read[i])",lw = 2)
	# end
	plot!(e_gains,mean(surv_times_q[:,2,:],dims=2), yerror = std(surv_times_q[:,2,:],dims = 2)./sqrt(n_ep), label = "R agent,ϵ = $(ϵs_read[2])",lw = 3)
	plot!(e_gains,mean(surv_times_q[:,3,:],dims=2), yerror = std(surv_times_q[:,3,:],dims = 2)./sqrt(n_ep), label = "R agent,ϵ = $(ϵs_read[3])",lw = 2,color = palette(:default)[4])
	plot!(e_gains,mean(surv_times_q[:,4,:],dims=2), yerror = std(surv_times_q[:,4,:],dims = 2)./sqrt(n_ep), label = "Random walker",lw = 2, color = palette(:default)[3])
	plot!(scale = :log,legend_position = :right, grid = false, fg_color_legend = :white,margin = 4Plots.mm, size = (500,350))
	#savefig("survival_rooms_paper.pdf")
	#plot!(surv_times_h[1,:],st = :histogram)
end

# ╔═╡ abb61da7-3711-4fa6-a021-2bb916411019
begin
	plot(ylabel = "Fraction occupied arena",xlabel = "Food gain")#, xlim = (2,14),xticks = collect(2:10))
	occ_h = 1 .-((1 .-(h_s_occ)).*(env1.sizex*env1.sizey-length(env1.obstacles)))./(env1.sizex*env1.sizey-33)
	plot!(e_gains,mean(occ_h,dims=2),yerror = std(occ_h,dims = 2)./(sqrt(n_ep)), label = "H agent",lw = 2)
	# for i in indices_to_plot
	# 	occ_q =q_s_occ[:,i,:]
	# 	plot!(e_gains,mean(occ_q,dims=2),yerror = std(occ_q,dims = 2)./(sqrt(n_ep)), label = "ϵ = $(ϵs_read[i])",c = cols[i],lw = 2)
	# end
	#(sqrt(n_ep))
	plot!(e_gains,mean(q_s_occ[:,2,:],dims=2),yerror = std(q_s_occ[:,2,:],dims = 2)./(sqrt(n_ep)), label = "R agent, ϵ = 0.45",lw = 2)
	plot!(e_gains,mean(q_s_occ[:,4,:],dims=2),yerror = std(q_s_occ[:,4,:],dims = 2)./(sqrt(n_ep)), label = "Random walker",lw = 2)
	#hline!([0], label = false, color = :black,ls = :dash)
	plot!(xscale = :log,legend_position = :bottomright, fg_color_legend = :white,grid = false, size = (600,400), margin = 4Plots.mm,ylim= (-0.09,1.05))
	#savefig("occupations_rooms_paper.pdf")
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

# ╔═╡ 631088a4-d06e-41c3-bc39-17db0ed9c9b8
begin
	#This computes occupancies running n_episodes, each with a max duration of t_episode, for both Q and H agent. It takes up to 8 hours for t_episode = 1E6 and 50 episodes.
	#h_s_occ,h_u_occ,q_s_occ,q_u_occ,times_h,times_q = occupancies_qh(energy_gains,n_episodes,t_episode,pars)
end

# ╔═╡ f059814e-6426-40f2-85b9-a6037e802928
energies = collect(2:10)

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
	draw_environment([3],[3],[u_0],env1)
	#savefig("arena.pdf")
end

# ╔═╡ 350b2469-5bdc-44d2-8f3e-fb80f749ac90
max_t = 50000

# ╔═╡ 78a5caf6-eced-4783-b950-26563f632be2
# begin
# 	@bind max_t2 PlutoUI.Select(["50000" => "short episode", "1E5" => "long episode"])
# end

# ╔═╡ 4a868ec2-b636-4d5d-a248-0a4e0cca3668
begin
	h_xpos_50,h_ypos_50,h_us_50,h_allx_50,h_ally_50,h_urgency_50,h_visited_s,h_visited_u,h_time = sample_trajectory(s_0,u_0,h_value_50,max_t,env1,false)
	#h_xpos_200,h_ypos_200,h_us_200,h_allx_200,h_ally_200,h_urgency_200 = sample_trajectory(s_0,u_0,h_value_200,max_t,env2)
end

# ╔═╡ 6edb5b48-b590-492d-bd3e-6a4f549aae30
begin
	q_xpos_50,q_ypos_50,q_us_50,q_allx_50,q_ally_50,q_visited_s,q_visited_u,q_time = sample_trajectory_q(s_0,u_0,q_value_50,ϵ,max_t,env1,false)
	#q_xpos_200,q_ypos_200,q_us_200,q_allx_200,q_ally_200 = sample_trajectory_q(s_0,u_0,q_value_200,max_t,env2)
end;

# ╔═╡ 4c51afa2-e294-4922-9cae-d087582d771c
function many_episodes_histogram(s_0,u_0,h_value,q_value,ϵ,max_t,n_episodes,env::environment)
	hs_x = Any[]
	hs_y = Any[]
	qs_x = Any[]
	qs_y = Any[]
	rand_x = Any[]
	rand_y = Any[]
	times_h = Any[]
	times_q = Any[]
	times_rand = Any[]
	for j in 1:n_episodes
		println("ep = ", j)
		h_xpos,h_ypos,h_us,h_allx,h_ally,h_urgency,h_visited_s,h_visited_u,h_time = sample_trajectory(s_0,u_0,h_value,max_t,env1,false)
		q_xpos,q_ypos,q_us,q_allx,q_ally,q_visited_s,q_visited_u,q_time = sample_trajectory_q(s_0,u_0,q_value,ϵ,max_t,env1,false)
		rand_xpos,rand_ypos,rand_us,rand_allx,rand_ally,rand_visited_s,rand_visited_u,rand_time = sample_trajectory_random(s_0,u_0,max_t,env1,false)
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
	end
	hs_x,hs_y,qs_x,qs_y,rand_x,rand_y,times_h,times_q,times_rand
end

# ╔═╡ c842a822-462b-4849-86f1-3fc717c492c1
n_episodes_hist = 1

# ╔═╡ 42423e46-2446-4cdd-84c1-fe824e5eb3c4
begin
	import Random
	Random.seed!(121)
	hs_x,hs_y,qs_x,qs_y,rand_x,rand_y,times_h,times_q,times_rand = many_episodes_histogram(s_0,u_0,h_value_50,q_value_50,ϵ,max_t,n_episodes_hist,env1)
end

# ╔═╡ 8cc41c87-4347-4deb-a4f0-7c70ac75ceec
begin
	Random.seed!(2)
	h_xpos_anim,h_ypos_anim,h_us_anim,h_allx_anim,h_ally_anim,h_urgency_anim = sample_trajectory(s_0,u_0,h_value,max_t_anim,env1)
end

# ╔═╡ a0729563-0b6d-4014-b8c7-9eb284a34606
if movie
	anim_h = animation(h_xpos_anim,h_ypos_anim,h_us_anim,max_t_anim,env1,color = palette(:default)[1])
end

# ╔═╡ 11b5409c-9db8-4b34-a111-7a62fedd23be
gif(anim_h, fps = 12)

# ╔═╡ bc83dd1c-b2f2-48ca-b994-3cb8e77e6ea2
begin
	Random.seed!(2)
	q_xpos_anim,q_ypos_anim,q_us_anim,q_allx_anim,q_ally_anim= sample_trajectory_q(s_0,u_0,q_value,ϵ,max_t_anim,env1)
end

# ╔═╡ 787bbe73-6052-41e0-bc8c-955e4a884886
if movie_q
	anim_q = animation(q_xpos_anim,q_ypos_anim,q_us_anim,max_t_anim,env1, title = "R agent, ϵ = $(ϵ)", color = palette(:default)[2])
	#gif(anim_q, "q_agent.gif", fps = 8)
end

# ╔═╡ 139c806d-3f52-4fb9-9fe8-c57259ed1b6f
gif(anim_q, fps = 12)

# ╔═╡ a5f43388-9e45-497f-b0f8-7f2987b3102d
anim_both = animation2(h_xpos_anim,h_ypos_anim,h_us_anim,q_xpos_anim,q_ypos_anim,q_us_anim,max_t_anim,env1)

# ╔═╡ 293c129d-dba4-4b04-aa0a-66e1b570aad4
gif(anim_both, fps = 10)

# ╔═╡ feaefaa1-c3ba-47d3-8fb2-6ba56d089001
rand_xpos_50,rand_ypos_50,rand_us_50,rand_allx_50,rand_ally_50,rand_visited_s,rand_visited_u,rand_time = sample_trajectory_random(s_0,u_0,max_t,env1,false)

# ╔═╡ df345648-a76d-44a6-b13f-9eee97cbbd5e
rand_time

# ╔═╡ 567f6b5d-c67e-4a43-9699-5625f1cc21a4
md"### Histogram of visited external states"

# ╔═╡ c933c84c-f158-4626-850b-7f5a164ea4aa
function plot_histogram(h_allx,h_ally,q_allx,q_ally,rand_allx,rand_ally,env1,t_hist = 500000,maxclim = 0.2)
	## H agent
	p_h_hist = plot(ticks = false, title = "H agent")#,ylabel = "\$u_{max} = 50\$")
	#Paint the whole arena 
	heatmap!(p_h_hist,zeros(env1.sizex,env1.sizey))
	#Draw obstacles
	plot!(p_h_hist,env1.obstaclesx,env1.obstaclesy,bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)), st = :histogram2d, color = "black")
	#Draw histogram
		histogram2d!(p_h_hist,h_allx,h_ally, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :true,cbar = false,clim = (0,maxclim))

	## R agent
	p_q_hist = plot(ticks = false, title = "R agent, ϵ = $(ϵ)")
	#Paint the whole arena
	heatmap!(p_q_hist,zeros(env1.sizex,env1.sizey))
	#Draw obstacles
	plot!(p_q_hist,env1.obstaclesx,env1.obstaclesy,bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)), st = :histogram2d, color = "black")
	#Draw histogram
	histogram2d!(p_q_hist, q_allx,q_ally, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :true,cbar = false,clim = (0,maxclim))

	## Random walker
	p_rand_hist = plot(ticks = false, title = "Random walker")
	#Paint the whole arena
	heatmap!(p_rand_hist,zeros(env1.sizex,env1.sizey))
	#Draw obstacles
	plot!(p_rand_hist,env1.obstaclesx,env1.obstaclesy,bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)), st = :histogram2d, color = "black")
	#Draw histogram
	histogram2d!(p_rand_hist, rand_allx,rand_ally, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = :true,clim = (0,maxclim))
	
	#Colorbar
	plot!(p_rand_hist,colorbar_ticks = collect(maxclim/4:maxclim/4:maxclim))
	# p_h200_hist = plot(ticks = false)
	# #plot!(p_hist1,env1.obstaclesx,env1.obstaclesy, st = :histogram2d)
	# heatmap!(p_h200_hist,zeros(env1.sizex,env1.sizey))
	# histogram2d!(p_h200_hist,h_allx_200,h_ally_200, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = true,clim = (0,0.2),cbar = false,ylabel = "\$u_{max} = 200\$")
	# p_q200_hist = plot(ticks = false)
	# heatmap!(p_q200_hist,zeros(env1.sizex,env1.sizey))
	# histogram2d!(p_q200_hist, q_allx_200,q_ally_200, bins = (collect(0.5:1:env1.sizex + 0.5),collect(0.5:1:env1.sizey + 0.5)),normalize = true,clim = (0,0.2))
	plot(p_h_hist,p_q_hist,p_rand_hist, layout = Plots.grid(1, 3, widths=[0.3,0.3,0.4]),size = (900,300),margin = 5Plots.mm)
	end

# ╔═╡ e671cb3e-2d1a-4196-9274-89d41ac323c8
begin
	plot_histogram(hs_x,hs_y,qs_x,qs_y,rand_x,rand_y,env1,Int(h_time),0.016)
	#savefig("locations_histogram_threeagents_nepisodes_$(n_episodes_hist)_gain_$(food_gain).pdf") 
end

# ╔═╡ b2918c10-ca06-4c1d-91c1-c17e4dd49c9d
md"### Animations for histograms? $(@bind animations CheckBox(default = false))"

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
	p_q50_hist = plot(ticks = false, title = "R agent, ϵ = $(ϵ)")
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
		maxclim = 3000
		tstep = 1000
		maxt = max_t
		hist_animated = animate_histogram(h_allx_50,h_ally_50,q_allx_50,q_ally_50,env1,maxclim,tstep,maxt)
	end
end

# ╔═╡ cc3af52b-1d47-48b5-bbc4-9ef1327f4dfa
gif(hist_animated,fps = 5,"histograms_animated.gif")

# ╔═╡ 1ace6a77-cc48-4b3d-8774-035015ffd74a
function animate_trajectory(h_x_pos,h_y_pos,q_x_pos,q_y_pos,env,tstep = 1,max_t = 500000)
	
	## H agent
	p_h50 = plot(ticks = false, title = "H agent")#,ylabel = "\$u_{max} = 50\$")
	p_q50 = plot(ticks = false, title = "R agent, ϵ = $(ϵ)")
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
		tstep_traj = 1000
		maxt_traj = max_t
		traj_animated = animate_trajectory(h_allx_50,h_ally_50,q_allx_50,q_ally_50,env1,tstep_traj,maxt_traj)
	end
end

# ╔═╡ cda6ab20-3dbb-43eb-9ae3-71889cb3da88
gif(traj_animated,fps = 5,"trajectories.gif")

# ╔═╡ 1483bfe5-150d-40f5-b9dc-9488dcbc88b2
md"### Histogram of visitation of internal states"

# ╔═╡ d93ef435-c460-40cc-96e7-817e9eaace55
begin
	p_us = plot(xlim = (0,100), xlabel = "Internal energy", ylabel = "Probability",legend_foreground_color = nothing, margin = 4Plots.mm, majorgrid = false,size = (400,300),minorgrid = false)
	bd = 1
	plot!(p_us,h_us_50, label = "H agent", bandwidth = bd, st = :density,linewidth = 3)
	plot!(p_us,q_us_50, label = "R agent", bandwidth = bd, st = :density,linewidth = 3)
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
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
ZipFile = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"

[compat]
Distributions = "~0.25.58"
Parameters = "~0.12.3"
Plots = "~1.29.0"
PlutoUI = "~0.7.38"
StatsPlots = "~0.14.34"
ZipFile = "~0.9.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

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
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "63d1e802de0c4882c00aee5cb16f9dd4d6d7c59c"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.1"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cc1a8e22627f33c789ab60b36a9132ac050bbf75"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.12"

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
git-tree-sha1 = "8a6b49396a4058771c5c072239b2e0a76e2e898c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.58"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

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
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "af237c08bda486b74318c8070adb96efa6952530"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

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
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

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
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

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
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

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
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

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
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "6d019f5a0465522bbfdd68ecfad7f86b535d6935"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.0"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded92de95031d4a8c61dfb6ba9adb6f1d8016ddd"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "aee446d0b3d5764e35289762f6a18e8ea041a592"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.11.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

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

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "027185efff6be268abbaf30cfd53ca9b59e3c857"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.10"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d457f881ea56bbfa18222642de51e0abf67b9027"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

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
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

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
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

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

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

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
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "ca9f8a0c9f2e41431dc5b7697058a3f8f8b89498"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.0"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "43a316e07ae612c461fd874740aeef396c60f5f8"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.34"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "e75d82493681dfd884a357952bbd7ab0608e1dc3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.7"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

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
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

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
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

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
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

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
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

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

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

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
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═a649da15-0815-438d-9bef-02c6d204656e
# ╠═25c56490-3b9c-4825-b91d-8b9e41fc0f6b
# ╠═422493c5-8a90-4e70-bd06-40f8e6b254f1
# ╠═76f77726-7776-4975-9f30-3887f13ae3e7
# ╠═393eaf2d-e8fe-4675-a7e6-32d0fe9ac4e7
# ╠═b4e7b585-261c-4044-87cc-cbf669768145
# ╟─7feeec1a-7d7b-4220-917d-049f1e9b101b
# ╠═7e68e560-45d8-4429-8bff-3a8229c8c84e
# ╟─194e91cb-b619-4908-aebd-3136107175b7
# ╟─a46ced5b-2e58-40b2-8eb6-b4840043c055
# ╟─9404080e-a52c-42f7-9abd-ea488bf7abc2
# ╟─0dcdad0b-7acc-4fc4-93aa-f6eacc077cd3
# ╟─0ce119b1-e269-41e2-80b7-991cae37cf5f
# ╟─8675158f-97fb-4222-a32b-49ce4f6f1d41
# ╟─92bca36b-1dc9-4c03-88c0-6a684dfbec9f
# ╟─c96e3331-1dcd-4b9c-b28d-d74493c8934d
# ╟─d0a5c0fe-895f-42d8-9db6-3b0fcc6bb43e
# ╠═155056b5-21ea-40d7-8cce-19fde5a1b150
# ╟─6c716ad4-23c4-46f8-ba77-340029fcce87
# ╠═8f2fdc23-1b82-4479-afe7-8eaf3304a122
# ╟─403a06a7-e30f-4aa4-ade1-55dee37cd514
# ╠═bd16a66c-9c2f-449c-a792-1073c54e990b
# ╟─ac3a4aa3-1edf-467e-9a47-9f6d6655cd04
# ╟─c6870051-0241-4cef-9e5b-bc876a3894fa
# ╟─d88e0e27-2354-43ad-9c26-cdc90beeea0f
# ╟─184636e2-c87d-4a89-b231-ff4aef8424d5
# ╠═82fbe5a0-34a5-44c7-bdcb-36d16f09ea7b
# ╠═a11b198f-0a55-4529-b44c-270f37ef773a
# ╠═e67db730-ca7c-4ef4-a2d2-7e001d5f7a79
# ╟─73722c01-adee-4bfd-97b4-60f2ced23725
# ╟─76f506dc-b21d-4e13-a8e8-9d1b3bd21b30
# ╟─aa5e5bf6-6504-4c01-bb36-df0d7306f9de
# ╟─ef9e78e2-d61f-4940-9e62-40c6d060353b
# ╠═8cc41c87-4347-4deb-a4f0-7c70ac75ceec
# ╟─a4457d71-27dc-4c93-81ff-f21b2dfed41d
# ╟─7ad00e90-3431-4e61-9a7f-efbc14d0724e
# ╟─b072360a-6646-4d6d-90ea-716085c53f66
# ╠═a0729563-0b6d-4014-b8c7-9eb284a34606
# ╠═11b5409c-9db8-4b34-a111-7a62fedd23be
# ╟─f98d6ea0-9d98-4940-907c-455397158f3b
# ╟─5f4234f5-fc0e-4cdd-93ea-99b6463b2ba1
# ╟─7a0173ac-240d-4f93-b413-45c6af0f4011
# ╠═27011f44-929a-4899-b822-539d270959e1
# ╠═caadeb3b-0938-4559-8122-348c960a6eb1
# ╠═29a4d235-8b03-4701-af89-cd289f212e7d
# ╠═819c1be2-339f-4c37-b8a3-9d8cb6be6496
# ╠═358bc5ca-c1f6-40f1-ba2d-7e8466531903
# ╟─40d62df0-53bb-4b46-91b7-78ffd621a519
# ╟─005720d3-5920-476b-9f96-39971f512452
# ╟─2379dcc3-53cb-4fb6-b1e8-c851e36acd1f
# ╠═47cf89c9-bb93-498d-9881-f079399b98ba
# ╠═bc83dd1c-b2f2-48ca-b994-3cb8e77e6ea2
# ╟─6e7b6b2a-5489-4860-930e-47b7df014840
# ╟─2ed5904d-03a3-4999-a949-415d0cf47328
# ╠═787bbe73-6052-41e0-bc8c-955e4a884886
# ╠═139c806d-3f52-4fb9-9fe8-c57259ed1b6f
# ╟─6a29cc32-6abf-41c1-b6e3-f4cb33b76f46
# ╠═a5f43388-9e45-497f-b0f8-7f2987b3102d
# ╠═293c129d-dba4-4b04-aa0a-66e1b570aad4
# ╟─5b1ba4f6-37a9-4e61-b6cc-3d495aa67c9d
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
# ╠═631088a4-d06e-41c3-bc39-17db0ed9c9b8
# ╠═f059814e-6426-40f2-85b9-a6037e802928
# ╟─1f3be0d8-d296-4eb0-a951-bd862914ae92
# ╠═bb45134a-88b1-40d2-a486-c7afe8ac744e
# ╠═f1d5ee65-10f5-424a-b018-2aff1a5d7ff8
# ╠═350b2469-5bdc-44d2-8f3e-fb80f749ac90
# ╠═78a5caf6-eced-4783-b950-26563f632be2
# ╠═4a868ec2-b636-4d5d-a248-0a4e0cca3668
# ╠═6edb5b48-b590-492d-bd3e-6a4f549aae30
# ╟─4c51afa2-e294-4922-9cae-d087582d771c
# ╠═c842a822-462b-4849-86f1-3fc717c492c1
# ╠═42423e46-2446-4cdd-84c1-fe824e5eb3c4
# ╠═feaefaa1-c3ba-47d3-8fb2-6ba56d089001
# ╠═df345648-a76d-44a6-b13f-9eee97cbbd5e
# ╟─567f6b5d-c67e-4a43-9699-5625f1cc21a4
# ╟─c933c84c-f158-4626-850b-7f5a164ea4aa
# ╠═e671cb3e-2d1a-4196-9274-89d41ac323c8
# ╟─b2918c10-ca06-4c1d-91c1-c17e4dd49c9d
# ╟─3e8a7dbb-8dfa-44f8-beab-ea32f3d478b4
# ╠═298d4faf-bca5-49a5-b0e8-9e0b3600e3ae
# ╠═cc3af52b-1d47-48b5-bbc4-9ef1327f4dfa
# ╟─1ace6a77-cc48-4b3d-8774-035015ffd74a
# ╠═19b1a5c0-c5a8-45bc-8cf0-fc6f97ccff91
# ╠═cda6ab20-3dbb-43eb-9ae3-71889cb3da88
# ╟─1483bfe5-150d-40f5-b9dc-9488dcbc88b2
# ╠═d93ef435-c460-40cc-96e7-817e9eaace55
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
