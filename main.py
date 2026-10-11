from fastapi import FastAPI, HTTPException,status
from pydantic import BaseModel
from typing import Optional, List, Dict, Optional
app = FastAPI()
#Schema for simulation:
class Simulation(BaseModel):
    workload_type: str
    cache_capacity: int
    trace_length: int
    policy: str
    random_seed: Optional[int] = 0

#Schema for comparison:
class Compare(BaseModel):
    cache_capacity: int
    workload_type: str
    trace_length: int
    policy: List[str]
    random_seed: Optional[int] = 0

@app.get("/")
def read_root():
    return {"Project": "Smartcache-ai", "status": "Running"}

#Health check endpoint:
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status":"Ok"}

#Policies endpoint:
@app.get("/policies", status_code=status.HTTP_200_OK)
def get_policies():
    return {"policies": ["LRU", "LFU", "Belady's Algorithm", "Smartcache Policy"]}

#Workloads endpoint:
@app.get("/workloads", status_code=status.HTTP_200_OK)
def get_workloads():
    return {"workloads": ["Looping hot set", "Sequential scan", 
    "Zipf-like skew", "Phase shift", "Mixed workload"]}

#Single simulation endpoint:
@app.post("/simulations",status_code=status.HTTP_200_OK)
def run_simulation(simulation: Simulation):
    simulation_dict = simulation.model_dump()
    if simulation_dict["cache_capacity"] <= 0 or simulation_dict["trace_length"] <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                             detail="Cache capacity and trace length must be positive integers")

    return{"message": "Simulation completed successfully",
           "configuration": simulation_dict,
            "simulation_result": {"hit_rate": 0.85, "miss_rate": 0.15,
             "total_requests": 1000, "cache_hits": 850, 
         "cache_misses": 150, "evictions": 50}
            }

#Comparison endpoint
@app.post("/simulations/compare", status_code=status.HTTP_200_OK)
def compare_simulations(compare_request: Compare):
      compare_request_dict = compare_request.model_dump()
      if compare_request_dict["cache_capacity"] <= 0 or compare_request_dict["trace_length"] <= 0:
          raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                              detail="Cache capacity and trace length must be positive integers")
      
      return{"message": "Comparison successful",
             "configuration": compare_request_dict,
                "comparison_results": {
                    "LRU": {"hit_rate": 0.80, "miss_rate": 0.20},
                    "LFU": {"hit_rate": 0.75, "miss_rate": 0.25},
                    "Belady's Algorithm": {"hit_rate": 0.90, "miss_rate": 0.10},
                    "Smartcache Policy": {"hit_rate": 0.85, "miss_rate": 0.15}
                }
            }