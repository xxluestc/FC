from fc_power.power_allocation.mpc_allocator import choose
def allocate(current_demand,*args,**kwargs):return choose([current_demand],*args,**kwargs)

