import h3

h = '8a0983d5250ffff'
print({
    'resolution': h3.h3_get_resolution(h),
    'base_cell': h3.h3_get_base_cell(h),
    'center': h3.h3_to_geo(h),
    'boundary': h3.h3_to_geo_boundary(h)
})