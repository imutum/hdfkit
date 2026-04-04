from mtmhdf.grid2dreader import Grid2DReader

if __name__ == "__main__":
    
    grid2dreader = Grid2DReader("tests\data\MOD021KM_L.1000.2015001004500.H27V04.000000.nc", do_grid_surround=True)
    print(grid2dreader.keys())
    dp = grid2dreader.read("/GeometricCorrection/DataSet_1000_5")
    print(data:=dp[-1:1202, -1:1202])
    print(data:=dp[:])
    print(data.shape)