import os
import cx_Oracle
import pandas as pd

class DB_Connection():
    def __init__(self):
        os.putenv('NLS_LANG', '.UTF8')
        # 연결에 필요한 기본 정보 (유저, 비밀번호, 데이터베이스 서버 주소)
        self.connection = cx_Oracle.connect("Ai_team1", "1234", "192.168.0.12:1521/orcl")
        self.cursor = self.connection.cursor()
        self.connection.autocommit = True

    def select_user(self):
        self.cursor.execute("select customer_id from CUSTOMERS where enter = 'True'")
        cnt = self.cursor.fetchone()
        return cnt[0]

    def get_username(self,cid): 
        self.cursor.execute("SELECT customer_name FROM customers WHERE customer_id=:cid",{"cid":cid})
        name = self.cursor.fetchone()
        return name[0]

# find product id based on product name
    def select_product(self,product_name):
        self.cursor.execute("SELECT product_id FROM products WHERE product_name = :name",{"name":product_name})
        pname = self.cursor.fetchone()
        return pname[0]

    # def get_productname (self,product_id):
    #     self.cursor.execute("SELECT product_name FROM products WHERE product_id = :pid", {"pid": product_id})
    #     name = self.cursor.fetchone()
    #     return name[0]

    def insert_cart(self, customer_id, product_id, qty):
        self.cursor.execute("INSERT INTO carts('cart_id, customer_id,product_id, cart_stock, cart_in) VALUES (cart_seq.nextval, :cid, :pid,:qty, systimestamp)",{"cid":customer_id,"pid":product_id,"qty":qty})


    def update_cart(self, customer_id, product_id, qty):
        self.cursor.execute("UPDATE carts SET cart_stock = :qty WHERE customer_id = :cid AND WHERE product_id = :pid)",{"cid":customer_id,"pid":product_id,"qty":qty})


#######################################################################

    def update_user(self, cnt):
        self.cursor.execute("update customers set enter = 'False' where  customer_id = :customer_id", {"customer_id":cnt})

    def update_login_session_T(self, customer_id):
        # 고객이 로그아웃 후 login_session을 리셋해 줌.
        self.cursor.execute("update customers set login_session = 'True', enter = 'True' where customer_id = :id", {"id": customer_id})

    def insert_check_In_Time(self, customer_id):
        # 고객이 로그인한 시간을 기
        self.cursor.execute("insert into in_and_out(in_out_id, customer_id, check_in) values(in_out_seq.nextval, :id, systimestamp)",{"id": customer_id})

    def update_login_session_F(self, customer_id):
        # 고객이 로그아웃 후 login_session을 리셋해 줌.
        self.cursor.execute("update customers set login_session = 'False' where customer_id = :id", {"id": customer_id})

    def insert_check_Out_Time(self, customer_id):
        # 고객이 로그인한 시간을 기록
        self.cursor.execute("update in_and_out set check_out = systimestamp where customer_id = :id and check_out is null",{"id": customer_id})

    def select_cart_product(self, customer_id):
        print('customer_id1:',customer_id)
        self.cursor.execute("""SELECT ROW_NUMBER() OVER (order by c.cart_id desc) as num, c.product_id, c.cart_in, p.product_name, p.product_price, c.cart_stock FROM carts c inner join Products p on c.product_id = p.product_id where c.customer_id = :id""", {"id": customer_id})
        cnta = self.cursor.fetchall()
        print('cnta',cnta)
        df = pd.DataFrame(cnta)
        return df

    def insert_order(self, customer_id, total_price):
        self.cursor.execute("""INSERT INTO orders(order_id, customer_id, total_price) VALUES (order_seq.nextval,:id,:total_price)"""
                       , {"id": customer_id, "total_price": total_price})

    def insert_order_detail(self, df):
        for index, row in df.iterrows():  # cart_id, product_id, cart_in, product_name, product_price, cart_stock
            print(row[0], row[1], row[2], row[3], row[4], row[5])
            self.cursor.execute("""INSERT INTO order_details (order_detail_id, order_id, product_id, cart_in, ordered_price, cart_stock) VALUES(order_details_seq.nextval, order_seq.currval, :product_id, :cart_in, :ordered_price, :cart_stock)""" , {"product_id": row[1], "cart_in": row[2], "ordered_price": row[4], "cart_stock": row[5]})
            # 상품목록에서 수량 감소j
            self.cursor.execute("select product_stock from products where product_id = :product_id", {"product_id": row[1]})
            cnt = self.cursor.fetchone()
            print('product_stock1', cnt[0])
            upd_stock = cnt[0] - row[5]
            print('upd_stock',upd_stock)
            self.cursor.execute("update products set product_stock = :upd_stock where product_id = :product_id",{"upd_stock":upd_stock, "product_id": row[1]})
            print('cursor',self.cursor)

    def delete_cart(self, customer_id):
        print('delete')
        self.cursor.execute("""delete carts where customer_id = :id""", {"id": customer_id})

db = DB_Connection()
db.select_user()