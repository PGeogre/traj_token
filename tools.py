# import h3

# h = '8a0983d5250ffff'
# print({
#     'resolution': h3.h3_get_resolution(h),
#     'base_cell': h3.h3_get_base_cell(h),
#     'center': h3.h3_to_geo(h),
#     'boundary': h3.h3_to_geo_boundary(h)
# })


import h3

h = "8a1f334164effff"

# 分辨率
res = h3.get_resolution(h)

# 基底单元
base = h3.h3_get_base_cell(h)

# 中心点经纬度（lat, lng）
lat, lng = h3.h3_to_geo(h)

# 六边形边界（经纬度坐标序列）
boundary = h3.h3_to_geo_boundary(h)

print("res:", res)                # 10
print("base cell:", base)         # 0..121 之间的一个整数
print("center:", lat, lng)        # 该六边形的大致中心点
print("boundary:", boundary)      # 6 个（或在五边形畸变附近可能是更多）顶点的经纬度