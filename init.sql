create table benchmarkresult
(
	id int auto_increment,
	dataset varchar(255) not null,
	algo varchar(255) not null,
	error_rate double not null,
	repetition int not null,
	moment_saved bigint not null,
	date_string varchar(100) not null,
	constraint benchmarkresult_pk
		primary key (id)
);