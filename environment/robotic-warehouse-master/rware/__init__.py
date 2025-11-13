from gymnasium import register

# 注册FulfillmentWarehouse环境
register(
    id="FulfillmentWarehouse-v0",
    entry_point="rware.fulfillment_warehouse:FulfillmentWarehouseEnv",
    kwargs={
        "num_agents": 32,
        "use_energy": False,
        "render_mode": None,
    },
)
