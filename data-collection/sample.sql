/*
Collection of training and testing sample
*/

-- Table for keeping sample
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='sample' AND xtype='U')
  CREATE TABLE guest.sample (
    valID INT NOT NULL,
    cntrID INT NOT NULL,
    supID INT NOT NULL,
    orgID INT NOT NULL,
    okpdID INT NOT NULL,
    cntr_reg_num VARCHAR(19),
    
    -- Supplier
    sup_cntr_num INT,
    sup_running_cntr_num INT,
    sup_good_cntr_num FLOAT,
    sup_fed_cntr_num FLOAT, 
    sup_sub_cntr_num FLOAT, 
    sup_mun_cntr_num FLOAT,
    sup_cntr_avg_price BIGINT,
    sup_cntr_avg_penalty_share FLOAT,
    sup_no_pnl_share FLOAT,
    sup_1s_sev FLOAT,
    sup_1s_org_sev FLOAT,
    sup_okpd_cntr_num INT,
    sup_sim_price_share FLOAT,
    
    -- Заказчик
    org_cntr_num INT,
    org_running_cntr_num INT,
    org_good_cntr_num FLOAT,
    org_fed_cntr_num FLOAT,
    org_sub_cntr_num FLOAT,
    org_mun_cntr_num FLOAT,
    org_cntr_avg_price BIGINT,
    org_1s_sev FLOAT,
    org_1s_sup_sev FLOAT,
    org_sim_price_share FLOAT,
    cntr_num_together INT,
    org_type INT,
    
    -- Russian classifier of products by type of economic activity
    okpd_cntr_num INT,
    okpd_good_cntr_num INT,
    okpd VARCHAR(9),

    -- Contract
    price BIGINT,
    pmp BIGINT,
    cntr_lvl INT,
    sign_date INT,
    exec_date INT,
    purch_type INT,
    price_higher_pmp BIT,
    price_too_low BIT,

    -- Target variable
    cntr_result BIT,
    
     -- Primary key
    PRIMARY KEY (valID, cntrID, supID, orgID, okpdID)
  )
GO

-- Ignore inserts with dublicated primary keys
ALTER TABLE guest.sample REBUILD WITH (IGNORE_DUP_KEY = ON)
GO

-- Insert bad contracts
INSERT INTO guest.sample
SELECT
val.ID, 
cntr.ID,
val.RefSupplier,
org.ID,
okpd.ID,
cntr.RegNum,

-- Supplier
guest.sup_stats.sup_cntr_num,
guest.sup_stats.sup_running_cntr_num,
guest.sup_stats.sup_good_cntr_num AS 'sup_good_cntr_num',
guest.sup_stats.sup_fed_cntr_num AS 'sup_fed_cntr_num',
guest.sup_stats.sup_sub_cntr_num AS 'sup_sub_cntr_num',
guest.sup_stats.sup_mun_cntr_num AS 'sup_mun_cntr_num',
guest.sup_stats.sup_cntr_avg_price,
guest.sup_stats.sup_cntr_avg_penalty,
guest.sup_stats.sup_no_pnl_share,
guest.sup_stats.sup_1s_sev,
guest.sup_stats.sup_1s_org_sev,
guest.okpd_sup_stats.cntr_num AS 'sup_okpd_cntr_num',
NULL, --Field, which will be calculated later

-- Customer
guest.org_stats.org_cntr_num,
guest.org_stats.org_running_cntr_num,
guest.org_stats.org_good_cntr_num AS 'org_good_cntr_num',
guest.org_stats.org_fed_cntr_num AS 'org_fed_cntr_num',
guest.org_stats.org_sub_cntr_num AS 'org_sub_cntr_num',
guest.org_stats.org_mun_cntr_num AS 'org_mun_cntr_num',
guest.org_stats.org_cntr_avg_price,
guest.org_stats.org_1s_sev,
guest.org_stats.org_1s_sup_sev,
NULL, --Field, which will be calculated later
guest.sup_org_stats.cntr_num AS 'cntr_num_together',
org.RefTypeOrg AS 'org_type',

-- Russian classifier of products by type of economic activity
guest.okpd_stats.good_cntr_num as 'okpd_good_cntr_num',
guest.okpd_stats.cntr_num AS 'okpd_cntr_num',
okpd.Code AS 'okpd', 

-- Contract
val.Price AS 'price',
val.PMP AS 'pmp',
val.RefLevelOrder AS 'cntr_lvl',
cntr.RefSignDate AS 'sign_date',
cntr.RefExecution AS 'exec_date',
cntr.RefTypePurch AS 'purch_type',
CASE WHEN (val.PMP > 0) AND (val.Price > val.PMP) THEN 1 ELSE 0 END AS 'price_higher_pmp',
CASE WHEN val.Price <= val.PMP * 0.6 THEN 1 ELSE 0 END AS 'price_too_low',

-- Target variable
guest.cntr_stats.result AS 'cntr_result'

FROM DV.f_OOS_Value AS val
INNER JOIN DV.d_OOS_Org AS org ON org.ID = val.RefOrg
INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
INNER JOIN DV.f_OOS_Product AS prod ON prod.RefContract = cntr.ID
INNER JOIN DV.d_OOS_Products AS prods ON prods.ID = prod.RefProduct
INNER JOIN DV.d_OOS_OKPD2 AS okpd ON okpd.ID = prods.RefOKPD2
INNER JOIN guest.sup_stats ON val.RefSupplier = guest.sup_stats.SupID
INNER JOIN guest.org_stats ON org.ID = guest.org_stats.OrgID
INNER JOIN guest.okpd_stats ON okpd.ID = guest.okpd_stats.OkpdID
INNER JOIN guest.okpd_sup_stats ON (guest.okpd_sup_stats.SupID = val.RefSupplier AND guest.okpd_sup_stats.OkpdID = okpd.ID)
INNER JOIN guest.sup_org_stats ON (guest.sup_org_stats.SupID = val.RefSupplier AND guest.sup_org_stats.OrgID = org.ID)
INNER JOIN guest.cntr_stats ON guest.cntr_stats.CntrID = cntr.ID
WHERE
	val.Price > 0 AND --Contract with positive price (real contract)
	cntr.RefStage IN (3, 4) AND --Contract is completed
	cntr.RefSignDate > guest.utils_get_init_year() AND --Contract is signed not earlier than <starting_point>
	cntr_stats.result = 1 --Contract is bad
GO

-- Insert good contracts
INSERT INTO guest.sample
SELECT TOP(CAST(@@ROWCOUNT*1.5 AS INT))
val.ID, 
cntr.ID,
val.RefSupplier,
org.ID,
okpd.ID,
cntr.RegNum,

-- Supplier
guest.sup_stats.sup_cntr_num,
guest.sup_stats.sup_running_cntr_num,
guest.sup_stats.sup_good_cntr_num AS 'sup_good_cntr_num',
guest.sup_stats.sup_fed_cntr_num AS 'sup_fed_cntr_num',
guest.sup_stats.sup_sub_cntr_num AS 'sup_sub_cntr_num',
guest.sup_stats.sup_mun_cntr_num AS 'sup_mun_cntr_num',
guest.sup_stats.sup_cntr_avg_price,
guest.sup_stats.sup_cntr_avg_penalty,
guest.sup_stats.sup_no_pnl_share,
guest.sup_stats.sup_1s_sev,
guest.sup_stats.sup_1s_org_sev,
guest.okpd_sup_stats.cntr_num AS 'sup_okpd_cntr_num',
NULL, --Field, which will be calculated later

-- Customer
guest.org_stats.org_cntr_num,
guest.org_stats.org_running_cntr_num,
guest.org_stats.org_good_cntr_num AS 'org_good_cntr_num',
guest.org_stats.org_fed_cntr_num AS 'org_fed_cntr_num',
guest.org_stats.org_sub_cntr_num AS 'org_sub_cntr_num',
guest.org_stats.org_mun_cntr_num AS 'org_mun_cntr_num',
guest.org_stats.org_cntr_avg_price,
guest.org_stats.org_1s_sev,
guest.org_stats.org_1s_sup_sev,
NULL, --Field, which will be calculated later
guest.sup_org_stats.cntr_num AS 'cntr_num_together',
org.RefTypeOrg AS 'org_type',

-- Russian classifier of products by type of economic activity
guest.okpd_stats.good_cntr_num as 'okpd_good_cntr_num',
guest.okpd_stats.cntr_num AS 'okpd_cntr_num',
okpd.Code AS 'okpd', 

-- Contract
val.Price AS 'price',
val.PMP AS 'pmp',
val.RefLevelOrder AS 'cntr_lvl',
cntr.RefSignDate AS 'sign_date',
cntr.RefExecution AS 'exec_date',
cntr.RefTypePurch AS 'purch_type',
CASE WHEN (val.PMP > 0) AND (val.Price > val.PMP) THEN 1 ELSE 0 END AS 'price_higher_pmp',
CASE WHEN val.Price <= val.PMP * 0.6 THEN 1 ELSE 0 END AS 'price_too_low',

-- Target variable
guest.cntr_stats.result AS 'cntr_result'

FROM DV.f_OOS_Value AS val
INNER JOIN DV.d_OOS_Org AS org ON org.ID = val.RefOrg
INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
INNER JOIN DV.f_OOS_Product AS prod ON prod.RefContract = cntr.ID
INNER JOIN DV.d_OOS_Products AS prods ON prods.ID = prod.RefProduct
INNER JOIN DV.d_OOS_OKPD2 AS okpd ON okpd.ID = prods.RefOKPD2
INNER JOIN guest.sup_stats ON val.RefSupplier = guest.sup_stats.SupID
INNER JOIN guest.org_stats ON org.ID = guest.org_stats.OrgID
INNER JOIN guest.okpd_stats ON okpd.ID = guest.okpd_stats.OkpdID
INNER JOIN guest.okpd_sup_stats ON (guest.okpd_sup_stats.SupID = val.RefSupplier AND guest.okpd_sup_stats.OkpdID = okpd.ID)
INNER JOIN guest.sup_org_stats ON (guest.sup_org_stats.SupID = val.RefSupplier AND guest.sup_org_stats.OrgID = org.ID)
INNER JOIN guest.cntr_stats ON guest.cntr_stats.CntrID = cntr.ID
WHERE
  val.Price > 0 AND --Contract with positive price (real contract)
  cntr.RefStage IN (3, 4) AND --Contract is finished
  cntr.RefSignDate > guest.utils_get_init_year() AND --Contract is signed not earlier than <starting_point>
  guest.cntr_stats.result = 0 --Contract is good
ORDER BY NEWID()
GO

-- Insert info about unfinished contracts (contracts to predict result)
INSERT INTO guest.sample
SELECT
val.ID, 
cntr.ID,
val.RefSupplier,
org.ID,
okpd.ID,
cntr.RegNum,

-- Supplier
guest.sup_stats.sup_cntr_num,
guest.sup_stats.sup_running_cntr_num,
guest.sup_stats.sup_good_cntr_num AS 'sup_good_cntr_num',
guest.sup_stats.sup_fed_cntr_num AS 'sup_fed_cntr_num',
guest.sup_stats.sup_sub_cntr_num AS 'sup_sub_cntr_num',
guest.sup_stats.sup_mun_cntr_num AS 'sup_mun_cntr_num',
guest.sup_stats.sup_cntr_avg_price,
guest.sup_stats.sup_cntr_avg_penalty,
guest.sup_stats.sup_no_pnl_share,
guest.sup_stats.sup_1s_sev,
guest.sup_stats.sup_1s_org_sev,
guest.okpd_sup_stats.cntr_num AS 'sup_okpd_cntr_num',
NULL,

-- Customer
guest.org_stats.org_cntr_num,
guest.org_stats.org_running_cntr_num,
guest.org_stats.org_good_cntr_num AS 'org_good_cntr_num',
guest.org_stats.org_fed_cntr_num AS 'org_fed_cntr_num',
guest.org_stats.org_sub_cntr_num AS 'org_sub_cntr_num',
guest.org_stats.org_mun_cntr_num AS 'org_mun_cntr_num',
guest.org_stats.org_cntr_avg_price,
guest.org_stats.org_1s_sev,
guest.org_stats.org_1s_sup_sev,
NULL,
guest.sup_org_stats.cntr_num AS 'cntr_num_together',
org.RefTypeOrg AS 'org_type',

-- Russian classifier of products by type of economic activity
guest.okpd_stats.good_cntr_num as 'okpd_good_cntr_num',
guest.okpd_stats.cntr_num AS 'okpd_cntr_num',
okpd.Code AS 'okpd', 

-- Contract
val.Price AS 'price',
val.PMP AS 'pmp',
val.RefLevelOrder AS 'cntr_lvl',
cntr.RefSignDate AS 'sign_date',
cntr.RefExecution AS 'exec_date',
cntr.RefTypePurch AS 'purch_type',
CASE WHEN (val.PMP > 0) AND (val.Price > val.PMP) THEN 1 ELSE 0 END AS 'price_higher_pmp',
CASE WHEN val.Price <= val.PMP * 0.6 THEN 1 ELSE 0 END AS 'price_too_low',

-- Target variable (not defined)
NULL

FROM DV.f_OOS_Value AS val
INNER JOIN DV.d_OOS_Org AS org ON org.ID = val.RefOrg
INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
INNER JOIN DV.f_OOS_Product AS prod ON prod.RefContract = cntr.ID
INNER JOIN DV.d_OOS_Products AS prods ON prods.ID = prod.RefProduct
INNER JOIN DV.d_OOS_OKPD2 AS okpd ON okpd.ID = prods.RefOKPD2
INNER JOIN guest.sup_stats ON val.RefSupplier = guest.sup_stats.SupID
INNER JOIN guest.org_stats ON org.ID = guest.org_stats.OrgID
INNER JOIN guest.okpd_stats ON okpd.ID = guest.okpd_stats.OkpdID
INNER JOIN guest.okpd_sup_stats ON (guest.okpd_sup_stats.SupID = val.RefSupplier AND guest.okpd_sup_stats.OkpdID = okpd.ID)
INNER JOIN guest.sup_org_stats ON (guest.sup_org_stats.SupID = val.RefSupplier AND guest.sup_org_stats.OrgID = org.ID)
WHERE
  val.Price > 0 AND --Contract with positive price (real contract)
  cntr.RefStage = 2 AND --Contract is running
  cntr.RefSignDate > guest.utils_get_init_year() --Contract is signed not earlier than <starting_point>
GO

-- Calculation of NULL field
UPDATE guest.sample
SET sup_sim_price_share = guest.sup_similar_contracts_by_price_share(val.RefSupplier, ss.sup_cntr_num, val.Price),
    org_sim_price_share = guest.org_similar_contracts_by_price_share(val.RefOrg, os.org_cntr_num, val.Price)
FROM guest.sample s
INNER JOIN DV.f_OOS_Value val ON s.valID = val.ID
INNER JOIN guest.sup_stats ss ON ss.SupID = val.RefSupplier
INNER JOIN guest.org_stats os ON os.OrgID = val.RefOrg
GO